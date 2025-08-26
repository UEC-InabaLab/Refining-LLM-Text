import os
import json
from tqdm import tqdm
from oss_llm import Llama_Swallow
from train_deberta import DebertaRegressionModel
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from safetensors.torch import load_file 
import torch


#　本番で使用する関数(baseline1とproposal)
def predict_score_with_deberta(dialogue_summary, recommend_sentence, rec_info):
    # モデルの事前学習済みの名前（Hugging Faceのモデルハブから）
    MODEL_NAME = "globis-university/deberta-v3-japanese-large"

    # 学習済みモデルの保存ディレクトリ（適宜変更してください）
    BEST_MODEL_DIR = "deberta_best_model_proposal&baseline1_new"
    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sep = tokenizer.sep_token
    # モデルの初期化
    best_model = DebertaRegressionModel(MODEL_NAME)

    # --- 3. モデルとトークナイザーのロード ---

    # safetensorsから状態辞書をロード
    model_path = os.path.join(BEST_MODEL_DIR, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    state_dict = load_file(model_path)
    best_model.load_state_dict(state_dict)

    # モデルを評価モードに設定
    best_model.eval()

    # --- 4. 入力データのトークン化 ---
    input_data = f"{dialogue_summary}{sep}{recommend_sentence}{sep}{rec_info}"
    # トークン化（単一のデータなのでリストにする）
    inputs = tokenizer(
        input_data,
        padding=True,         # パディングを有効にする
        truncation=True,      # トランケーションを有効にする
        max_length=1024,       # 必要に応じて最大長を設定
        return_tensors="pt",
        return_token_type_ids=False  # PyTorchのテンソルとして返す
    )

    # --- 5. デバイスの設定 ---

    # GPUが利用可能であればGPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # --- 6. 予測の実行 ---

    with torch.no_grad():
        # モデルに入力を渡して予測を取得
        outputs = best_model(**inputs)

        # 出力をCPUに移動し、NumPy配列に変換
        predictions = outputs["logits"].cpu().numpy()

    # --- 7. 予測結果の表示 ---

    # 必要に応じて次元を削減し、リストに変換
    predicted_score = predictions.squeeze().tolist()

    return predicted_score

def predict_score_with_deberta_baseline2(dialogue_summary, rec_info):
    # モデルの事前学習済みの名前（Hugging Faceのモデルハブから）
    MODEL_NAME = "globis-university/deberta-v3-japanese-large"

    # 学習済みモデルの保存ディレクトリ（適宜変更してください）
    BEST_MODEL_DIR = "deberta_best_model_baseline2_new"
    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sep = tokenizer.sep_token
    # モデルの初期化
    best_model = DebertaRegressionModel(MODEL_NAME)

    # --- 3. モデルとトークナイザーのロード ---

    # safetensorsから状態辞書をロード
    model_path = os.path.join(BEST_MODEL_DIR, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    state_dict = load_file(model_path)
    best_model.load_state_dict(state_dict)

    # モデルを評価モードに設定
    best_model.eval()

    # --- 4. 入力データのトークン化 ---
    input_data = f"{dialogue_summary}{sep}{rec_info}"
    # トークン化（単一のデータなのでリストにする）
    inputs = tokenizer(
        input_data,
        padding=True,         # パディングを有効にする
        truncation=True,      # トランケーションを有効にする
        max_length=1024,       # 必要に応じて最大長を設定
        return_tensors="pt",
        return_token_type_ids=False  # PyTorchのテンソルとして返す
    )

    # --- 5. デバイスの設定 ---

    # GPUが利用可能であればGPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # --- 6. 予測の実行 ---

    with torch.no_grad():
        # モデルに入力を渡して予測を取得
        outputs = best_model(**inputs)

        # 出力をCPUに移動し、NumPy配列に変換
        predictions = outputs["logits"].cpu().numpy()

    # --- 7. 予測結果の表示 ---

    # 必要に応じて次元を削減し、リストに変換
    predicted_score = predictions.squeeze().tolist()

    return predicted_score


# 動作確認で使用するための関数
def old_predict_score_with_deberta(input_data):
    # モデルの事前学習済みの名前（Hugging Faceのモデルハブから）
    MODEL_NAME = "globis-university/deberta-v3-japanese-large"

    # 学習済みモデルの保存ディレクトリ（適宜変更してください）
    BEST_MODEL_DIR = "deberta_baseline1&proposal"
    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sep = tokenizer.sep_token
    # モデルの初期化
    best_model = DebertaRegressionModel(MODEL_NAME)

    # --- 3. モデルとトークナイザーのロード ---

    # safetensorsから状態辞書をロード
    model_path = os.path.join(BEST_MODEL_DIR, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    state_dict = load_file(model_path)
    best_model.load_state_dict(state_dict)

    # モデルを評価モードに設定
    best_model.eval()

    # --- 4. 入力データのトークン化 ---
    # トークン化（単一のデータなのでリストにする）
    inputs = tokenizer(
        input_data,
        padding=True,         # パディングを有効にする
        truncation=True,      # トランケーションを有効にする
        max_length=1024,       # 必要に応じて最大長を設定
        return_tensors="pt",
        return_token_type_ids=False  # PyTorchのテンソルとして返す
    )

    # --- 5. デバイスの設定 ---

    # GPUが利用可能であればGPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # --- 6. 予測の実行 ---

    with torch.no_grad():
        # モデルに入力を渡して予測を取得
        outputs = best_model(**inputs)

        # 出力をCPUに移動し、NumPy配列に変換
        predictions = outputs["logits"].cpu().numpy()

    # --- 7. 予測結果の表示 ---

    # 必要に応じて次元を削減し、リストに変換
    predicted_score = predictions.squeeze().tolist()

    return predicted_score


if __name__=="__main__":
# 予測したい単一の入力データ
# data1 = f"50歳の女性2人組で秋に行われる旅行についての要件をまとめらゆと、以下のようなことが言える。1.行先：奈良県２.時期：秋３.人数：２人４.趣味：おちゃん５.目的：紅葉を見たいと思います。{sep}腕利き料理長の技が冴える、ホテル内の日本料理店。昼なら一人前ごとに小鍋で出される大和鍋御膳2400円を。鶏がらだしに豆乳と牛乳を加えたスープで、大和肉鶏や大和芋の揚げ団子を煮込んだまろやかな味は病みつきになるほど。所要30～60分くらい 英語メニューあり 1000～3000円くらい（昼食） 5000～1万円くらい（夕食） 全席禁煙 ひとりにおすすめ デートにおすすめ 記念日におすすめ 接待におすすめ。{sep}ホテル内にあり、腕の良い料理人がいる日本食レストランです。ランチタイムには、一人前ずつ小さな土瓶に入れて出てくる「和風だしおでん」がオススメです。おいしいおでんと一緒に、ゆったりとした時間を過ごすことができます。また、ディナータイムは、接待や記念日にぴったりなコース料理が用意されています。ぜひ、訪れてみてください。"
# data2 =  f"50歳の女性2人組で秋に行われる旅行についての要件をまとめらゆと、以下のようなことが言える。1.行先：奈良県２.時期：秋３.人数：２人４.趣味：おちゃん５.目的：紅葉を見たいと思います。{sep}四季折々に美しい、大和路の風景を表現した料理が味わえる店。昼のおすすめは、松花堂1600円、茶がゆ膳2100円、奈良三昧3240円の3種類。いずれもコーヒーと、お菓子付き。季節の味が盛り込まれている。夜は奈良の膳3240円、ミニ懐石5400円、ともに予約制の奈良づくし7560円と特別懐石1万800円の4種類。所要60～90分くらい 女子おすすめ 英語メニューあり 1000～3000円くらい（昼食） 3000～5000円くらい（夕食） 全席禁煙 ひとりにおすすめ デートにおすすめ 記念日におすすめ 接待におすすめ 女子会におすすめ{sep}昔ながらの和食を楽しめます。料理は四つ切りの器に盛られていて見た目も綺麗です。店内は落ち着いた雰囲気でデートや記念日にぴったりです。また、接待や女子会にも利用できます。"
# data3 = f"65歳以上の3人家族で北海道を見て回りたいと思う。だけれど、寒い場所に長時間いるのは体に良くないかもなので、温かい食べ物を楽しむことにします。{sep}自然豊かな海と大地が育んだ、郷土の味わいをワインと共に気軽に楽しめるビストロ。日中は暖かな陽が差し込み、夜は照明を落とし落ち着いた空間になる店内は、気取らないアットホームな雰囲気で、ホテル内にありながらカジュアルに食事ができるのがうれしい。素材本来の味を活かしたメニューは、北海道の豊かな自然で育った「大地の糧」、「海の糧」を親しみやすい郷土料理に。約1500本のワインとマリアージュもおすすめだ。少人数での会食や女子会などのパーティーも可能。 英語メニューあり 1000～3000円くらい（昼食） 3000～5000円くらい（夕食） 全席禁煙 ひとりにおすすめ デートにおすすめ 記念日におすすめ 接待におすすめ 女子会におすすめ{sep}北の大地で生まれた食材をふんだんに使った料理を楽しめます。ホテル内のレストランなので、カフェのような気楽さがありますが、料理のクオリティは高く、素敵な時間を過ごすことができます。また、ワイナリーも併設されているので、お酒好きの方にもぴったりです。ぜひ一度足を運んでみてください。"
# data4 = "こんにちは"
# data5 =  f"##解答例Bは、歴史的建造物や美術館、博物館などを訪れることが好きで、特にヨーロッパの文化に興味を持っている。{sep}鳥海山を流れる月光川上流の渓谷沿いに、ブナの原生林を通り抜けるトレッキングコースが整備されている。春は新緑、夏は滝の清涼感、秋は紅葉と季節ごとに楽しめる。スタート地点の駐車場から一ノ滝までは歩いて5分ほど。付近には神社や展望台がある。さらに20分ほど登るとニノ滝に到着。ニノ滝は落差が約18mあり水量も多く、鳥海の名瀑として有名。所要30～60分くらい 女子おすすめ 夏におすすめ 秋におすすめ{sep}月の光が美しい月海川の上流にあり、春には新芽が生い茂り夏には清らかな水の流れが見られる。秋には美しく色づいた木々の葉を眺めながら散策することができ、紅く染まった景観はとても綺麗である。滞在時間は約30分から1時間ほどで，女性の方にもお勧めできるスポットである。"
# モデルの事前学習済みの名前（Hugging Faceのモデルハブから）
    dialogue1 = "沖縄旅行を計画している女性は、ビーチサンダルを履いて海を見たり、牛車に乗ったりしたいと考えている。また、アグー豚の料理も楽しみにしている。"
    dialogue2 = "沖縄旅行を計画しているBさんは、50代の独身女性です。彼女は、海でゆったりとした時間を楽しみたいと考えています。また、離れ島での宿泊も希望しています。さらに、牛が引く車に乗って海岸沿いを巡るツアーや、アグー豚を使った料理にも興味を持っています。AさんはBさんに、彼女が気に入りそうな美味しいお店やおすすめの場所を教えてくれています。"
    recommend_sentence2 = "Aランクの肉と、新鮮で美味しい海の幸が味わえるお店です。お祝い事や記念日のディナーにもぴったり。落ち着いた雰囲気の店内でゆっくりとお料理を堪能できます。リーズナブルな価格で本格的な料理が食べられるので、家族や友人と一緒に訪れてみてはいかがでしょうか。"
    recommend_sentence1 ="沖縄県本部町にあるレストランです。A級の牛肉を使用した料理や新鮮な魚介類を楽しめます。接待にもお勧めです。"
    rec_info = "和牛A5・A4等級のもとぶ牛はじめ、あぐー、山原若鶏、料理長自らの目ききで選ぶ鮮度抜群の海の幸等、とことん食材にこだわった料理をリーズナブルに楽しめる。所要30～60分くらい 英語メニューあり 3000～5000円くらい（夕食） 接待におすすめ"
    print(predict_score_with_deberta(dialogue1,recommend_sentence1,rec_info))
    print(predict_score_with_deberta(dialogue2,recommend_sentence2,rec_info))
    