import torch

# .bin ファイルを読み込む
training_args = torch.load("../src-chatrec/dpo-summary-results/training_args.bin")

# そのまま print しても OK ですが、
# 属性として格納されているので vars() を使うと辞書形式で見やすいです
for key, value in vars(training_args).items():
    print(f"{key}: {value}")