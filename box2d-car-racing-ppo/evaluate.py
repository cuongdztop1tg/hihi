import os

import gymnasium as gym
import numpy as np
import torch

from envwrapper import EnvWrapper
from model import ActorCritic
from util import save_gif

"""

def main():
    if not os.path.exists("./model"):
        print("ERROR: No model saved")
        exit(1)

    # Randomly generate 50 different seeds for env.reset to make sure that all models
    # are going through the same set of tracks
    rng = np.random.default_rng(seed=315)
    track_seeds = rng.choice(2**32 - 1, size=50, replace=False)

    env = EnvWrapper(
        gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
    )
    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0])

    highest_score = -1000
    best_avg_score = -1000
    best_fname = ""

    # Put all the checkpoint models through 50 test drives to evaluate performance
    for fname in sorted(os.listdir("./model")):
        if not fname.startswith("checkpoint_") or not fname.endswith(".pt"):
            continue

        print(f"Evaluating {fname} ... ", end="", flush=True)

        checkpoint = torch.load(f"./model/{fname}")
        model.load_state_dict(checkpoint["model"])

        avg_score = 0

        for seed in track_seeds:
            frames = []
            score = 0

            state, _ = env.reset(seed=seed.item())
            while True:
                frames.append(env.render())

                state_tensor = torch.tensor(
                    np.expand_dims(state, axis=0), dtype=torch.float32
                )
                with torch.no_grad():
                    action, _, _ = model(state_tensor, determinstic=True)
                    action = action.detach().cpu().numpy()[0]
                next_state, reward, terminated, truncated, _ = env.step(action)

                score += reward
                state = next_state
                if terminated or truncated:
                    break

            # Save the best play for demo
            if score > highest_score:
                highest_score = score
                save_gif(frames, "best_play.gif")

            avg_score += score

        print(f"Average Score = {avg_score / 50:.5f}")
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_fname = fname
            torch.save(checkpoint, "./model/model.pt")

    print(f"The best model is {best_fname}, with a average score of {best_avg_score / 50:5f}")
"""


def main():
    # --- Thiết lập cơ bản ---
    model_dir = "/kaggle/working/hihi/box2d-car-racing-ppo/model_a2c"  # <--- THAY ĐỔI 1: Đường dẫn thư mục checkpoint A2C
    best_model_path = os.path.join(
        model_dir, "best_a2c_model.pt"
    )  # <--- THAY ĐỔI 1: Tên file model tốt nhất
    env_id = "CarRacing-v3"  # <--- THAY ĐỔI 4: Đảm bảo khớp với training (v2 thường dùng hơn)
    eval_seed = 315  # Seed để tạo tập track đánh giá
    num_eval_tracks = 50  # Số lượng track để đánh giá mỗi checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    if not os.path.exists(model_dir):
        print(f"ERROR: Checkpoint directory not found at {model_dir}")
        exit(1)

    # --- Tạo tập track đánh giá cố định ---
    print(
        f"Generating {num_eval_tracks} evaluation track seeds using master seed {eval_seed}..."
    )
    rng = np.random.default_rng(seed=eval_seed)
    # Sử dụng integers thay vì choice để tránh phụ thuộc vào size của không gian seed quá lớn
    track_seeds = rng.integers(low=0, high=2**31 - 1, size=num_eval_tracks)

    # --- Khởi tạo môi trường và mô hình ---
    # Sử dụng render_mode='rgb_array' vì chúng ta cần frame để lưu GIF
    env = EnvWrapper(gym.make(env_id, domain_randomize=False, render_mode="human"))
    # Khởi tạo model với kiến trúc giống lúc train, sau đó chuyển sang device
    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0]).to(
        device
    )
    model.eval()  # Chuyển model sang chế độ đánh giá (quan trọng cho dropout, batchnorm nếu có)

    highest_score_ever = -float("inf")  # Điểm cao nhất trên một track bất kỳ
    best_avg_score = -float("inf")  # Điểm trung bình cao nhất của một model
    best_fname = ""

    print(f"Evaluating checkpoints in {model_dir}...")

    # --- Vòng lặp đánh giá các checkpoint ---
    for fname in sorted(os.listdir(model_dir)):
        # Chỉ xử lý các file checkpoint hợp lệ
        # if not fname.startswith("a2c_checkpoint_") or not fname.endswith(".pt"):
        #     continue

        if fname != "best_a2c_model.pt":
            continue

        ckpt_path = os.path.join(model_dir, fname)
        print(f"  Evaluating {fname} ... ", end="", flush=True)

        try:
            # Tải checkpoint, đảm bảo tải lên đúng device
            checkpoint = torch.load(
                ckpt_path, map_location=device
            )  # <--- THAY ĐỔI 3: map_location
            # Tải trọng số vào model (đã ở trên device)
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )  # <--- THAY ĐỔI 2: Sử dụng đúng key
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue  # Bỏ qua checkpoint này nếu không tải được

        current_model_total_score = 0
        num_valid_runs = 0

        # --- Chạy model trên các track đánh giá ---
        for i, seed in enumerate(track_seeds):
            frames = []
            score = 0
            terminated = False
            truncated = False

            try:
                state, _ = env.reset(
                    seed=seed.item()
                )  # Cần seed.item() vì seed là numpy int
                step = 0
                max_eval_steps = (
                    1000  # Giới hạn số bước tối đa cho mỗi lần chạy đánh giá
                )
                while not terminated and not truncated and step < max_eval_steps:
                    # Luôn render frame để có thể lưu GIF nếu cần
                    current_frame = env.render()
                    if current_frame is not None:
                        frames.append(current_frame)

                    state_tensor = torch.tensor(
                        np.expand_dims(state, axis=0),
                        dtype=torch.float32,
                        device=device,  # <--- THAY ĐỔI 3: Chuyển tensor sang device
                    )
                    with torch.no_grad():
                        # Sử dụng chế độ deterministic=True để đánh giá
                        action, _, _ = model(state_tensor, determinstic=True)
                        action_np = action.cpu().numpy()[
                            0
                        ]  # Chuyển action về CPU/numpy

                    next_state, reward, terminated, truncated, _ = env.step(action_np)

                    score += reward
                    state = next_state
                    step += 1

                # Chỉ tính điểm nếu episode chạy ít nhất 1 bước
                if step > 0:
                    current_model_total_score += score
                    num_valid_runs += 1

                    # Lưu lại lần chạy có điểm cao nhất từng thấy
                    if score > highest_score_ever:
                        print(
                            f"\n    New highest single score: {score:.2f} from {fname} on seed {seed}"
                        )
                        highest_score_ever = score
                        try:
                            # Giả sử hàm save_gif tồn tại và nhận list các frame numpy
                            save_gif(frames, "best_play.gif")
                            print(f"    Saved best play GIF.")
                        except NameError:
                            print("    (save_gif function not found, cannot save GIF)")
                        except Exception as e:
                            print(f"    (Error saving GIF: {e})")

            except Exception as e:
                print(
                    f"\n    Error during evaluation run with seed {seed} for {fname}: {e}"
                )
                # Có thể bỏ qua lần chạy này hoặc xử lý khác

        # --- Tính điểm trung bình và cập nhật model tốt nhất ---
        if num_valid_runs > 0:
            avg_score = current_model_total_score / num_valid_runs
            print(
                f"Average Score = {avg_score:.3f} ({num_valid_runs}/{num_eval_tracks} valid runs)"
            )

            if avg_score > best_avg_score:
                print(
                    f"    >>> New best average score! Saving {fname} as {best_model_path}"
                )
                best_avg_score = avg_score
                best_fname = fname
                # Lưu lại toàn bộ checkpoint của model tốt nhất (bao gồm cả optimizer state nếu muốn)
                torch.save(
                    checkpoint, best_model_path
                )  # <--- THAY ĐỔI 1: Lưu vào đường dẫn mới
        else:
            print("No valid runs completed for this checkpoint.")

    env.close()  # Đóng môi trường sau khi đánh giá xong

    if best_fname:
        print("-" * 30)
        print(f"Evaluation finished.")
        print(f"The best model checkpoint is: {best_fname}")
        print(f"With an average score of:   {best_avg_score:.3f}")
        print(f"Saved as:                   {best_model_path}")
        print(
            f"Highest single track score found: {highest_score_ever:.2f} (best_play.gif)"
        )
    else:
        print("-" * 30)
        print(
            "Evaluation finished, but no valid checkpoints were successfully evaluated."
        )


if __name__ == "__main__":
    main()
