import torch
import gymnasium as gym
import numpy as np
import os
import time  # Để thêm delay nhỏ nếu cần
import argparse

# Import các lớp và hàm cần thiết từ các file của bạn
# Đảm bảo các file này nằm trong cùng thư mục hoặc trong PYTHONPATH
try:
    from envwrapper import EnvWrapper
    from model import ActorCritic

    # Import fix_random_seeds nếu bạn muốn dùng seed cố định
    # from util import fix_random_seeds
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Hãy đảm bảo các file envwrapper.py và model.py ở đúng vị trí.")
    exit()


def run_live_demo(checkpoint_path, seed=None, max_steps=1000):
    """
    Chạy demo trực tiếp trong cửa sổ Pygame.

    Args:
        checkpoint_path (str): Đường dẫn đến file checkpoint .pt.
        seed (int, optional): Random seed cho môi trường demo.
        max_steps (int): Số bước tối đa cho một episode demo.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Lỗi: Không tìm thấy checkpoint tại '{checkpoint_path}'")
        return

    # if seed is not None:
    #     print(f"Fixing random seed: {seed}")
    #     try:
    #         fix_random_seeds(seed) # Sử dụng hàm này nếu có
    #     except NameError:
    #         print("Hàm fix_random_seeds không tồn tại, chỉ đặt seed cho môi trường.")
    #         # Có thể thêm đặt seed numpy/torch ở đây nếu cần
    #         # np.random.seed(seed)
    #         # torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    env = None  # Khởi tạo để đảm bảo có thể gọi close trong finally
    try:
        # --- 1. Khởi tạo môi trường với render_mode="human" ---
        print("Khởi tạo môi trường CarRacing-v3 (render_mode='human')...")
        # Chế độ "human" sẽ mở cửa sổ Pygame
        env_raw = gym.make("CarRacing-v3", render_mode="human", domain_randomize=False)

        # --- Áp dụng Wrapper giống như lúc huấn luyện ---
        # Điều này QUAN TRỌNG để đảm bảo observation space khớp với model
        print("Áp dụng EnvWrapper...")
        # Truyền seed vào EnvWrapper nếu wrapper của bạn sử dụng nó
        env = EnvWrapper(env_raw, seed=seed)
        input_shape = env.observation_space.shape
        # Kiểm tra xem action space có phải là Box không
        if isinstance(env.action_space, gym.spaces.Box):
            n_actions = env.action_space.shape[0]
        else:
            # Xử lý trường hợp action space khác nếu cần
            print(
                f"Kiểu Action Space không được hỗ trợ cho demo này: {type(env.action_space)}"
            )
            return
        print(
            f"Input shape (sau wrapper): {input_shape}, Số chiều actions: {n_actions}"
        )

        # --- 2. Khởi tạo mô hình ---
        print("Khởi tạo mô hình ActorCritic...")
        model = ActorCritic(input_shape, n_actions).to(device)

        # --- 3. Tải checkpoint ---
        print(f"Tải checkpoint từ: {checkpoint_path}")
        # Load lên device hiện tại để tránh lỗi mismatched device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Tải state dict thành công.")
            # Lấy thông tin bổ sung từ checkpoint nếu có
            episode = checkpoint.get("episode", "N/A")
            loss = checkpoint.get("loss", "N/A")
            print(f"Checkpoint được lưu từ episode: {episode}, Loss: {loss}")
        except KeyError:
            print(
                "Lỗi: Checkpoint không chứa 'model_state_dict'. Đang thử tải trực tiếp..."
            )
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print("Tải state_dict trực tiếp thành công.")
            except Exception as e_load:
                print(f"Lỗi khi tải state_dict trực tiếp: {e_load}")
                return
        except Exception as e:
            print(f"Lỗi không xác định khi tải checkpoint: {e}")
            return

        # --- Đặt model vào chế độ đánh giá ---
        model.eval()
        print("Model đặt ở chế độ eval().")

        # --- 4. Chạy vòng lặp demo ---
        print("\nBắt đầu Demo (Nhấn Ctrl+C trong terminal để dừng)...")
        # Reset môi trường, truyền seed nếu có
        state, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        step_count = 0

        while not terminated and not truncated and step_count < max_steps:
            # Chuẩn bị state cho model
            state_tensor = torch.tensor(
                np.expand_dims(state, axis=0), dtype=torch.float32, device=device
            )

            # Lấy action từ model (không tính gradient)
            with torch.no_grad():
                # Sử dụng deterministic=True để lấy action ổn định (mean của Gaussian)
                action_tensor, _, _ = model.forward(state_tensor, determinstic=True)
                action_np = action_tensor.cpu().numpy()[0]  # Chuyển về numpy array

            # Thực hiện action trong môi trường
            # Môi trường với render_mode="human" sẽ tự động render cửa sổ
            next_state, reward, terminated, truncated, info = env.step(action_np)

            # Cập nhật state và thông tin
            state = next_state
            total_reward += reward
            step_count += 1

            # print(f"Step: {step_count}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}") # Bỏ comment nếu muốn xem log từng step

            # Thêm một khoảng nghỉ nhỏ để dễ quan sát hơn (tùy chọn)
            # time.sleep(0.01)

            # Kiểm tra xem cửa sổ có bị đóng không (một số backend gym/pygame có thể trả về terminated/truncated)
            # Hoặc có thể cần kiểm tra sự kiện pygame nếu tích hợp sâu hơn, nhưng thường không cần thiết.

        print("\nDemo kết thúc.")
        print(f"Số bước đã thực hiện: {step_count}")
        print(f"Tổng phần thưởng đạt được: {total_reward:.2f}")
        if terminated:
            print("Lý do kết thúc: Terminated (Hoàn thành mục tiêu hoặc thất bại)")
        if truncated:
            print(f"Lý do kết thúc: Truncated (Đạt giới hạn {max_steps} bước)")

    except KeyboardInterrupt:
        print("\nDemo bị dừng bởi người dùng (Ctrl+C).")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình demo: {e}")
        import traceback

        traceback.print_exc()  # In chi tiết lỗi
    finally:
        # --- 5. Đóng môi trường ---
        if env is not None:
            print("Đóng môi trường...")
            env.close()
            print("Môi trường đã đóng.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chạy demo A2C trực tiếp trong cửa sổ Pygame."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="jerrykal githuub ppo/model_a2c/a2c_checkpoint_3000.pt",  # Đường dẫn mặc định
        help="Đường dẫn đến file checkpoint A2C (.pt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,  # Mặc định không dùng seed cố định
        help="Random seed cho môi trường demo (tùy chọn, để tái tạo)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,  # Giới hạn số bước như trong training
        help="Số bước tối đa cho mỗi episode demo",
    )
    args = parser.parse_args()

    run_live_demo(args.checkpoint, args.seed, args.steps)
