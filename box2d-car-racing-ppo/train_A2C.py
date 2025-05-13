import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym  # Đảm bảo đã import gymnasium
import os  # Đảm bảo đã import os
from torch.utils.tensorboard import SummaryWriter  # Đảm bảo đã import SummaryWriter
from envwrapper import EnvWrapper
from model import ActorCritic
from agent import A2C
from util import fix_random_seeds, play


def calculate_discounted_returns(b_rewards, discount_factor):
    """Tính toán discounted returns (G_t) từ một chuỗi phần thưởng."""
    discounted_returns = np.zeros_like(
        b_rewards, dtype=np.float32
    )  # Khởi tạo bằng float32
    running_add = 0
    for i in reversed(range(b_rewards.shape[0])):
        running_add = b_rewards[i] + discount_factor * running_add
        discounted_returns[i] = running_add
    return discounted_returns


def train(
    env,
    model,
    agent,
    device,
    n_episodes=3000,
    discount_factor=0.99,
    max_steps_per_episode=1000,
):
    """
    Hàm huấn luyện mô hình sử dụng agent (A2C).

    Args:
        env: Môi trường Gym đã được wrap.
        model: Mô hình ActorCritic.
        agent: Đối tượng agent (lớp A2C).
        device: Thiết bị tính toán ('cuda' hoặc 'cpu').
        n_episodes: Số lượng episode tối đa để huấn luyện.
        discount_factor: Hệ số chiết khấu gamma.
        max_steps_per_episode: Số bước tối đa cho mỗi episode.
    """
    if not os.path.exists("./model_a2c_test"):  # Đổi tên thư mục để tránh ghi đè PPO
        os.makedirs("./model_a2c_test")
    if not os.path.exists("./plot_a2c_test"):
        os.makedirs("./plot_a2c_test")

    # Sử dụng tên thư mục riêng cho A2C logs
    writer = SummaryWriter(log_dir="./plot_a2c_test")

    # Tính max_steps dựa trên frame skip nếu có, hoặc sử dụng giá trị cố định
    # max_steps = max_steps_per_episode // getattr(env, 'n_frames', 1) # Ước lượng nếu có n_frames
    max_steps = max_steps_per_episode  # Sử dụng giá trị cố định cho đơn giản

    print(f"Training on {device} using A2C...")
    print(f"Max steps per episode: {max_steps}")

    score_history = []
    avg_score_history = []
    best_avg_score = -np.inf

    for episode in range(n_episodes):
        # --- Thu thập dữ liệu từ một episode (Rollout) ---
        # Sử dụng list để lưu trữ vì không biết trước độ dài episode
        states_list = []
        actions_list = []
        action_probs_list = []  # Vẫn thu thập nhưng không dùng cho A2C learn
        state_values_list = []
        rewards_list = []
        terminated = False
        truncated = False

        state, _ = env.reset()
        current_step = 0

        while not terminated and not truncated and current_step < max_steps:
            state_tensor = torch.tensor(
                np.expand_dims(state, axis=0), dtype=torch.float32, device=device
            )
            # Lấy action, log_prob, value từ mô hình hiện tại
            with torch.no_grad():  # Không cần tính gradient khi thu thập dữ liệu
                action, action_prob, state_value = model.forward(state_tensor)

            action_np = action.cpu().numpy()[0]
            action_prob_np = action_prob.cpu().numpy()  # Log probability
            state_value_np = state_value.cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action_np)

            # Lưu trữ dữ liệu của bước này
            states_list.append(state)
            actions_list.append(action_np)
            action_probs_list.append(action_prob_np)
            state_values_list.append(state_value_np)
            rewards_list.append(reward)

            state = next_state
            current_step += 1
        # print(current_step)/
        # Độ dài thực tế của episode
        step_cnt = len(rewards_list)

        # Nếu episode kết thúc sớm, không có gì để làm thêm
        if step_cnt == 0:
            print(
                f"[Episode {episode + 1:4d}/{n_episodes}] Skipped - Episode ended immediately."
            )
            continue

        # --- Xử lý dữ liệu thu thập được ---
        b_states = np.array(states_list, dtype=np.float32)
        b_actions = np.array(actions_list, dtype=np.float32)
        # b_action_probs_np = np.array(action_probs_list, dtype=np.float32) # Không cần cho A2C learn
        b_state_values_np = np.array(state_values_list, dtype=np.float32)
        b_rewards_np = np.array(rewards_list, dtype=np.float32)

        # Chuyển đổi sang tensor
        b_states = torch.from_numpy(b_states).to(device)
        b_actions = torch.from_numpy(b_actions).to(device)
        b_state_values = torch.from_numpy(b_state_values_np).to(
            device
        )  # Cần để tính advantage

        # --- Tính toán Returns và Advantages ---
        b_returns_np = calculate_discounted_returns(b_rewards_np, discount_factor)
        b_returns = torch.from_numpy(b_returns_np.copy()).to(
            device, dtype=torch.float32
        )  # Chuyển returns sang tensor

        # Tính advantages: A(s, a) = G_t - V(s_t)
        # V(s_t) được lấy từ lúc thu thập dữ liệu (dùng mô hình cũ hơn một chút so với lúc cập nhật)
        b_advantages = b_returns - b_state_values

        # Chuẩn hóa advantages (quan trọng cho sự ổn định của A2C/PPO)
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )
        # Đảm bảo advantages không yêu cầu gradient vì chúng được coi là mục tiêu (target)
        b_advantages = b_advantages.detach()

        # Chuẩn hóa returns (thường cũng tốt cho Critic loss)
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
        b_returns = b_returns.detach()  # Returns cũng là target

        # --- Cập nhật tham số bằng A2C ---
        # Chỉ truyền những gì A2C.learn cần
        loss, actor_loss, critic_loss, entropy = agent.learn(
            b_states,
            b_actions,
            # b_action_probs, # LOẠI BỎ: A2C không cần log probs cũ
            b_returns,
            b_advantages,
        )

        # --- Logging và In kết quả ---
        total_reward = b_rewards_np.sum()

        score_history.append(total_reward)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            ckpt_path = f"./model_a2c_test/best_a2c.pt"
            print(f"Saving checkpoint to {ckpt_path}... ", flush=True)
            torch.save(
                {
                    "episode": episode + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),  # Lưu cả optimizer state
                    "loss": loss,
                },
                ckpt_path,
            )

        print(
            f"[Episode {episode + 1:4d}/{n_episodes}] Steps = {step_cnt}, Loss = {loss:.2f}, ",
            f"Actor Loss = {actor_loss:.2f}, Critic Loss = {critic_loss:.2f}, ",
            f"Entropy = {entropy:.2f}, ",
            f"Total Reward = {total_reward:.2f}, ",
            f"Avg score(100) = {avg_score:.2f}",
        )

        # Lưu vào TensorBoard
        writer.add_scalar("Loss/Episode", loss, episode + 1)
        writer.add_scalar("Loss/Actor Loss", actor_loss, episode + 1)
        writer.add_scalar("Loss/Critic Loss", critic_loss, episode + 1)
        writer.add_scalar("Performance/Entropy", entropy, episode + 1)
        writer.add_scalar("Performance/Total Reward", total_reward, episode + 1)
        writer.add_scalar("Performance/Episode Length", step_cnt, episode + 1)
        writer.flush()

        # Lưu checkpoint định kỳ
        if (episode + 1) % 50 == 0:
            ckpt_path = f"./model_a2c_test/a2c_checkpoint_{episode + 1:04d}.pt"
            print(f"Saving checkpoint to {ckpt_path}... ", flush=True)
            torch.save(
                {
                    "episode": episode + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),  # Lưu cả optimizer state
                    "loss": loss,
                },
                ckpt_path,
            )
            # Lưu video/gif nếu hàm play tồn tại
            try:
                play(model, f"Train_episode_{episode + 1:04d}.gif")
            except NameError:
                pass  # Bỏ qua nếu hàm play không tồn tại
            print("Done!")

    writer.close()
    print("Training finished.")


def main():
    # Fix random seeds (giả sử hàm này tồn tại)
    seed = 315
    fix_random_seeds(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Khởi tạo môi trường
    # Đảm bảo tên môi trường đúng và EnvWrapper được định nghĩa
    try:
        # Sử dụng render_mode='rgb_array' nếu không cần hiển thị cửa sổ
        env_raw = gym.make(
            "CarRacing-v2", domain_randomize=False, render_mode="rgb_array"
        )
        env = EnvWrapper(env_raw, seed=seed)  # Giả sử EnvWrapper đã định nghĩa
    except Exception as e:
        print(f"Lỗi khi tạo môi trường: {e}")
        print(
            "Hãy đảm bảo bạn đã cài đặt 'pip install gymnasium[box2d]' và tên môi trường là chính xác."
        )
        return

    # Khởi tạo mô hình (Giả sử ActorCritic đã định nghĩa)
    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0]).to(
        device
    )

    # --- Khởi tạo Agent là A2C ---
    lr = 1e-4  # Tốc độ học (có thể cần điều chỉnh)
    critic_weight = 0.5  # Trọng số Critic loss
    ent_weight = 0.01  # Trọng số Entropy bonus
    agent = A2C(
        model, learning_rate=lr, critic_weight=critic_weight, entropy_weight=ent_weight
    )
    print(
        f"Initialized A2C agent with lr={lr}, critic_w={critic_weight}, entropy_w={ent_weight}"
    )
    # -----------------------------

    # Bắt đầu huấn luyện
    try:
        train(
            env,
            model,
            agent,
            device,
            n_episodes=3000,
            discount_factor=0.99,
            max_steps_per_episode=1000,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()  # Luôn đóng môi trường


if __name__ == "__main__":
    # Cần import các lớp và hàm cần thiết trước khi chạy main
    # from your_module import EnvWrapper, DiagonalGaussian, ActorCritic, A2C, calculate_discounted_returns, fix_random_seeds, play
    main()
