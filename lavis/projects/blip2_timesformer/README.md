# BLIP-2 TimeSformer

BLIP-2 TimeSformer là mô hình video-language tích hợp mô hình backbone TimeSformer với kiến trúc BLIP-2 để tạo ra mô hình mạnh mẽ có khả năng mô tả nội dung video.

## Cài đặt

Trước khi sử dụng, hãy đảm bảo bạn đã cài đặt các gói phụ thuộc:

```bash
pip install -e .
pip install decord
```

## Checkpoints

Một số checkpoint quan trọng:

- Pretrained TimeSformer: Được huấn luyện trên Kinetics-400
- Pretrained BLIP-2: Sử dụng ViT-G/14 làm backbone cho phần thị giác và Q-Former cho phần đa phương thức
- Finetuned BLIP-2 TimeSformer: Checkpoint đã được fine-tune trên dataset video-caption

## Sử dụng

### Sinh mô tả cho một video (Inference)

Sử dụng script `inference.py` để tạo caption cho một video:

```bash
python lavis/projects/blip2_timesformer/inference.py \
    --video /đường/dẫn/đến/video.mp4 \
    --model_type caption \
    --checkpoint /đường/dẫn/đến/checkpoint.pth \
    --n_frames 8 \
    --image_size 224 \
    --num_beams 5 \
    --max_length 30 \
    --min_length 8 \
    --repetition_penalty 1.2 \
    --length_penalty 1.0 \
    --top_p 0.9 \
    --temperature 0.7
```

Các tham số quan trọng:
- `--video`: Đường dẫn đến file video cần tạo caption
- `--model_type`: Loại model (pretrain, caption,...)
- `--checkpoint`: Đường dẫn đến file checkpoint model
- `--n_frames`: Số frame trích xuất từ video (mặc định: 8)
- `--num_beams`: Số lượng beam cho beam search (mặc định: 5)
- `--repetition_penalty`: Hệ số phạt lặp lại (mặc định: 1.2)
- `--length_penalty`: Hệ số phạt độ dài (mặc định: 1.0) 
- `--temperature`: Nhiệt độ của sampling, giá trị thấp hơn sẽ tạo ra caption ít ngẫu nhiên hơn (mặc định: 0.7)
- `--use_nucleus_sampling`: Sử dụng nucleus sampling thay vì beam search

### Xử lý hàng loạt video (Batch Inference)

Sử dụng script `batch_inference.py` để tạo caption cho nhiều video cùng lúc:

```bash
python lavis/projects/blip2_timesformer/batch_inference.py \
    --video_dir /thư/mục/chứa/videos \
    --output_file /kết/quả/captions.json \
    --video_ext mp4 \
    --model_type caption \
    --checkpoint /đường/dẫn/đến/checkpoint.pth \
    --batch_size 4 \
    --n_frames 8 \
    --num_beams 5 \
    --repetition_penalty 1.2 \
    --temperature 0.7
```

Các tham số bổ sung:
- `--video_dir`: Thư mục chứa các file video cần xử lý
- `--output_file`: File đầu ra để lưu kết quả (định dạng JSON)
- `--video_ext`: Định dạng video cần xử lý (mặc định: mp4)
- `--batch_size`: Số lượng video xử lý cùng lúc (mặc định: 1)

### Đánh giá mô hình (Evaluation)

Sử dụng script đánh giá trên tập dữ liệu MSR-VTT:

```bash
bash lavis/projects/blip2_timesformer/eval/evaluate_msrvtt.sh
```

## Tối ưu hóa sinh caption

Để có được chất lượng caption tốt nhất, bạn có thể điều chỉnh các tham số sau:

- Sử dụng `temperature` thấp hơn (0.5-0.7) để caption chính xác hơn
- Tăng `repetition_penalty` (1.2-1.5) để tránh lặp từ
- Điều chỉnh `length_penalty` (1.0-1.5) để có caption dài hơn
- Tăng `num_beams` (5-10) để có nhiều lựa chọn hơn trong beam search

## Xử lý lỗi thường gặp

- **Lỗi CUDA Memory**: Giảm `batch_size` hoặc `n_frames`
- **Lỗi không tìm thấy checkpoint**: Kiểm tra lại đường dẫn đến file checkpoint
- **Lỗi video không đọc được**: Kiểm tra định dạng video và thử một thư viện khác như PyAV

## Trích dẫn

Nếu bạn sử dụng mã nguồn này, vui lòng trích dẫn:

```bibtex
@article{li2023blip2,
  title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  journal={arXiv preprint arXiv:2301.12597},
  year={2023}
}

@article{bertasius2021spacetime, 
  title={Is Space-Time Attention All You Need for Video Understanding?}, 
  author={Gedas Bertasius and Heng Wang and Lorenzo Torresani}, 
  journal={arXiv preprint arXiv:2102.05095}, 
  year={2021} 
}
``` 