import math

def validate_frame_size(frame_size: int):
    t = frame_size / 32.0
    return int(max(32, (math.floor(t) * 32)))

def validate_frame_count(duration: float, fps: int):
    frame_count = duration * fps
    t = (frame_count - 1.0) / 8.0
    return int(max(9, (math.ceil(t) * 8.0) + 1.0))

def validate_frame_size_2(frame_size: int) -> int:
    return max(32, (frame_size // 32) * 32)

def validate_frame_count_2(duration: float, fps: int) -> int:
    frame_count = duration * fps
    count_int = max(1, int(frame_count))
    ceil_steps = (count_int + 6) // 8
    return max(9, (ceil_steps * 8) + 1)

def main():
    print(validate_frame_size(129))
    print(validate_frame_size_2(129))
    print(validate_frame_count(5, 10))
    print(validate_frame_count_2(5, 10))

if __name__ == "__main__":
    main()