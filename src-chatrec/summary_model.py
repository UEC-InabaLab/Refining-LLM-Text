# -*- coding: utf-8 -*-
import logging
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
from logger import CustomLogger

custom_logger = CustomLogger(
    logger_name="my_module",
    log_level=logging.DEBUG,  # ログレベルの設定
    log_to_console=True,      # コンソールに出力するか
    log_to_file=False,         # ファイルに出力するか
    log_file_path="./logs"    # ログファイル保存先
)

# ロガーの取得
logger = custom_logger.get_logger()

# 1. Hugging Faceキャッシュディレクトリの設定
os.environ['HF_HOME'] = '../.venv/cache'  # カレントディレクトリ基準の相対パス

logging.getLogger("transformers").setLevel(logging.ERROR)

class Llama_Swallow:
    """
    A class to handle the loading and inference of the Llama-3.1-Swallow-8B model for text generation and summarization tasks.
    """

    def __init__(self, model_name: str = "./dpo-summary-results"):
        """
        Initializes the Llama_Swallow model by loading the tokenizer and model for causal language modeling. 
        Sets the device to 'cuda' if available, otherwise defaults to 'cpu'.
        
        Args:
            model_name: model path or model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            pad_token_id=self.tokenizer.pad_token_id
        )
        # print(self.model.config.max_position_embeddings)

    def __generate_text(self, prompt, temperature = 0.2):
        """
        Generates text based on the input prompt using the loaded language model.

        Args:
            prompt (str): The input text prompt to guide the text generation.

        Returns:
            str: Generated text based on the input prompt.
        """
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids_len = inputs["input_ids"].shape[1]
        
        # Perform inference to generate text
        with torch.no_grad():
                outputs = self.model.generate(
                    max_new_tokens=500,
                    inputs=inputs["input_ids"],
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=temperature, # 普段は0.2，　選好データ作成時は0.7に変更
                    top_k=0,
                    top_p=0.9,
                    early_stopping=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    num_beams=3,
                    do_sample=True
                )
        
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True)
        return generated_text.replace(" ", "").replace("\n", "")
    
    def is_japanese(self, char):
    # Unicodeの範囲を使用して判定
        ranges = [
            (0x3040, 0x309F),  # ひらがな
            (0x30A0, 0x30FF),  # カタカナ
            (0x4E00, 0x9FFF),  # 漢字
            (0x3400, 0x4DBF),  # 漢字拡張A
            (0xF900, 0xFAFF),  # 漢字互換
            (0x20000, 0x2A6DF), # 漢字拡張B
            (0x2A700, 0x2B73F), # 漢字拡張C
            (0x2B740, 0x2B81F), # 漢字拡張D
            (0x2B820, 0x2CEAF), # 漢字拡張E
            (0x2CEB0, 0x2EBEF), # 漢字拡張F
        ]
        code_point = ord(char)
        for start, end in ranges:
            if start <= code_point <= end:
                return True
        return False

    def calculate_japanese_ratio(self, text, exclude_punctuation=True):
        """
        文章中の日本語文字の割合を計算します。

        Args:
            text (str): 対象のテキスト。
            exclude_punctuation (bool): 句読点やスペースを除外するかどうか。

        Returns:
            float: 日本語文字の割合（0から100のパーセンテージ）。
        """
        if not text:
            return 0.0

        if exclude_punctuation:
            # 句読点やスペースなどの不要な文字を除外
            # 正規表現で日本語以外の文字を除外
            filtered_text = re.sub(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]', '', text)
        else:
            filtered_text = text

        total_chars = len(filtered_text)
        if total_chars == 0:
            return 0.0

        japanese_chars = sum(1 for char in filtered_text if self.is_japanese(char))
        ratio = (japanese_chars / total_chars)  # パーセンテージ表示

        return ratio

    def generate_summary(self, dialogue, user_name, temperature = 0.2):
        """
        Generates a summary for a given dialogue by processing the text with the model.

        Args:
            dialogue (str): The dialogue text to summarize.
            user_name(str): "A" or "B"

        Returns:
            str: Generated summary of the input dialogue.
        """
        A_prompt = f"""
あなたは高性能な分析アシスタントです。以下の対話履歴を分析して、ユーザAの嗜好・経験・趣味を抽出し、簡潔に要約してください。
1. タスク: 対話履歴からユーザAの嗜好・経験・趣味を抽出
2. 出力形式: 文章で出力
3. 抽出すべき情報
- 好きな活動（スポーツ、芸術、音楽鑑賞など）
- 好きな場所や訪問したい場所
- 好きな食べ物・飲み物
- 好きなエンターテイメント（映画、テレビ、ゲーム、音楽など）
- 興味のある物事
- 収集している物やコレクション
- 日常的に行っている楽しみな活動
4. 抽出してはいけない情報
- 性格特性（親切、几帳面など）
- コミュニケーションスタイル
- 価値観や信念
- 対人関係の特徴
- 感情表現のパターン
- 思考プロセスの特徴

--Example--
【対話履歴】
A:よろしくお願いします。\nB:こちらこそよろしくお願いいたします。今年のゴールデンウィークはお出かけされましたか。\nA:あいにく緊急事態宣言エリアだったので外出できませんでした。どちらかお出かけされましたか？\nB:私もちょっと買い物に出かけたくらいでほとんどうちにいました。去年もこんな感じのゴールデンウィークだったような気がします。\nA:そうですよね。コロナ以降なかなか外出は難しいですよね。\nB:去年の終わり頃は、来年になったら収束しそうな気がしていましたが、全然だめですね。\nA:もうしばらくはこのままでしょうかね。収束したらどんなことがしたいですか？\nB:思う存分旅行に行ったり、博物館や美術館に行きたいですね。\nA:旅行いいですね。国内ですか？それとも海外ですか？\nB:まずは国内旅行に行きたいです。空気と緑がきれいな場所でのんびりしたい気分です。\nA:素敵ですね。自然が豊かなところだと、とてもリフレッシュできそうですし。\nB:そうですね、外出自粛が続くと精神的につらいですよね。コロナ後に行きたい場所とかってございますか。\nA:自粛が長いので、ディズニーランドなどのテーマパークや音楽フェスに行ってワイワイしたいですね。\nB:ディズニーランドもいいですね。そういえば去年からずっと一度も行っていません。\nA:開園していましたが、なかなかチケットも取れなくて。早く普通に行けるようになる日が待ち遠しいです。\nB:ディズニーランドって定期的に行かないとなんだか落ち着かなくなりますよね。子供がとっても行きたがってます。\nA:お子さんもディズニーお好きなんですね。美女と野獣のアトラクションも新しくできましたし、また楽しめそうですよね。\nB:そうですね、一日も早く収束してもらいたいですね。テーマパークってどうしても密になってしまいますよね。\nA:そうですよね。もうしばらく先になるかもしれませんが、コロナが明けまで頑張りたいですね。\nB:ワクチン接種が早く進むといいなと思います。うちは母がようやく一回目の接種を終えました。\nA:そうでしたか。無事に接種の予約も取れてよかったですね。これから私たちも早く接種できるといいですね。\n"

【要約】
コロナ禍で緊急事態宣言エリアに住んでおり、外出を控えており、コロナ収束後はディズニーランドなどのテーマパークや音楽フェスに行きたいと考えている。ディズニーランドに行く習慣があるようだが、コロナ禍でチケット入手が難しい状況に悩んでいる。また、美女と野獣のアトラクションなど、ディズニーの新しい施設に興味を持っている。自然が豊かな場所での活動もリフレッシュできると考えている。ワクチン接種に前向きな姿勢を示している。

--Let's begin!--
【対話履歴】
{dialogue}

【ユーザAの要約】
                
"""
        B_prompt = f"""
あなたは高性能な分析アシスタントです。以下の対話履歴を分析して、ユーザーBの嗜好・経験・趣味を抽出し、簡潔に要約してください。
1. タスク: 対話履歴からユーザーBの嗜好・経験・趣味を抽出
2. 出力形式: 文章で出力
3. 抽出すべき情報
- 好きな活動（スポーツ、芸術、音楽鑑賞など）
- 好きな場所や訪問したい場所
- 好きな食べ物・飲み物
- 好きなエンターテイメント（映画、テレビ、ゲーム、音楽など）
- 興味のある物事
- 収集している物やコレクション
- 日常的に行っている楽しみな活動
4. 抽出してはいけない情報
- 性格特性（親切、几帳面など）
- コミュニケーションスタイル
- 価値観や信念
- 対人関係の特徴
- 感情表現のパターン
- 思考プロセスの特徴

--Example--
【対話履歴】
A:よろしくお願いします。\nB:こちらこそよろしくお願いいたします。今年のゴールデンウィークはお出かけされましたか。\nA:あいにく緊急事態宣言エリアだったので外出できませんでした。どちらかお出かけされましたか？\nB:私もちょっと買い物に出かけたくらいでほとんどうちにいました。去年もこんな感じのゴールデンウィークだったような気がします。\nA:そうですよね。コロナ以降なかなか外出は難しいですよね。\nB:去年の終わり頃は、来年になったら収束しそうな気がしていましたが、全然だめですね。\nA:もうしばらくはこのままでしょうかね。収束したらどんなことがしたいですか？\nB:思う存分旅行に行ったり、博物館や美術館に行きたいですね。\nA:旅行いいですね。国内ですか？それとも海外ですか？\nB:まずは国内旅行に行きたいです。空気と緑がきれいな場所でのんびりしたい気分です。\nA:素敵ですね。自然が豊かなところだと、とてもリフレッシュできそうですし。\nB:そうですね、外出自粛が続くと精神的につらいですよね。コロナ後に行きたい場所とかってございますか。\nA:自粛が長いので、ディズニーランドなどのテーマパークや音楽フェスに行ってワイワイしたいですね。\nB:ディズニーランドもいいですね。そういえば去年からずっと一度も行っていません。\nA:開園していましたが、なかなかチケットも取れなくて。早く普通に行けるようになる日が待ち遠しいです。\nB:ディズニーランドって定期的に行かないとなんだか落ち着かなくなりますよね。子供がとっても行きたがってます。\nA:お子さんもディズニーお好きなんですね。美女と野獣のアトラクションも新しくできましたし、また楽しめそうですよね。\nB:そうですね、一日も早く収束してもらいたいですね。テーマパークってどうしても密になってしまいますよね。\nA:そうですよね。もうしばらく先になるかもしれませんが、コロナが明けまで頑張りたいですね。\nB:ワクチン接種が早く進むといいなと思います。うちは母がようやく一回目の接種を終えました。\nA:そうでしたか。無事に接種の予約も取れてよかったですね。これから私たちも早く接種できるといいですね。\n"

【要約】
コロナ禍で主に自宅で過ごしており、買い物程度の外出しかしておらず、コロナ収束後は旅行や博物館・美術館訪問を希望している。特に国内旅行を優先し、空気と緑がきれいな自然豊かな場所でのんびりしたいと考えている。子供がいて、子供もディズニーランドを好んでおり、定期的にディズニーランドに行く習慣があるが、コロナ禍で一年以上行けていない状況。家族に高齢の母親がおり、母親は既にワクチン接種（1回目）を完了している。

--Let's begin!--
【対話履歴】
{dialogue}

【ユーザBの要約】
           
"""
        if user_name == "A":
            prompt = A_prompt
        elif user_name == "B":
            prompt = B_prompt
        else:
            raise ValueError(f"Invalid user_name: {user_name}. Expected 'A' or 'B'.")  
        for _ in range(5):
            generated_summary = self.__generate_text(prompt, temperature)
            if len(generated_summary)>=30 and self.calculate_japanese_ratio(generated_summary)>0.9:
                break
        return generated_summary

    def generate_summaries(self, dialogue, user_name, num_sentence=5):
        summaries_list = []
        for _ in range(num_sentence):
            summary = self.generate_summary(dialogue=dialogue, user_name=user_name, temperature=1.0)
            summaries_list.append(summary)
        return dialogue, summaries_list 
            

if __name__ == "__main__":
# 対話要約分生成の検証
    # llm = Llama_Swallow()
    # dialogue = "A:よろしくお願いします。\nB:こちらこそよろしくお願いいたします。今年のゴールデンウィークはお出かけされましたか。\nA:あいにく緊急事態宣言エリアだったので外出できませんでした。どちらかお出かけされましたか？\nB:私もちょっと買い物に出かけたくらいでほとんどうちにいました。去年もこんな感じのゴールデンウィークだったような気がします。\nA:そうですよね。コロナ以降なかなか外出は難しいですよね。\nB:去年の終わり頃は、来年になったら収束しそうな気がしていましたが、全然だめですね。\nA:もうしばらくはこのままでしょうかね。収束したらどんなことがしたいですか？\nB:思う存分旅行に行ったり、博物館や美術館に行きたいですね。\nA:旅行いいですね。国内ですか？それとも海外ですか？\nB:まずは国内旅行に行きたいです。空気と緑がきれいな場所でのんびりしたい気分です。\nA:素敵ですね。自然が豊かなところだと、とてもリフレッシュできそうですし。\nB:そうですね、外出自粛が続くと精神的につらいですよね。コロナ後に行きたい場所とかってございますか。\nA:自粛が長いので、ディズニーランドなどのテーマパークや音楽フェスに行ってワイワイしたいですね。\nB:ディズニーランドもいいですね。そういえば去年からずっと一度も行っていません。\nA:開園していましたが、なかなかチケットも取れなくて。早く普通に行けるようになる日が待ち遠しいです。\nB:ディズニーランドって定期的に行かないとなんだか落ち着かなくなりますよね。子供がとっても行きたがってます。\nA:お子さんもディズニーお好きなんですね。美女と野獣のアトラクションも新しくできましたし、また楽しめそうですよね。\nB:そうですね、一日も早く収束してもらいたいですね。テーマパークってどうしても密になってしまいますよね。\nA:そうですよね。もうしばらく先になるかもしれませんが、コロナが明けまで頑張りたいですね。\nB:ワクチン接種が早く進むといいなと思います。うちは母がようやく一回目の接種を終えました。\nA:そうでしたか。無事に接種の予約も取れてよかったですね。これから私たちも早く接種できるといいですね。\n"
    # dialogue2 = "A:何か趣味とか好きな物はありますか？\nB:一応音楽聞くのは好きです。\nA:音楽ですか。好きなミュージシャンとかいますか？\nB:今だったら古いですけどイギリスのロックバンドのジェネシスとかゲスの極み乙女。などです。\nA:イギリスのロックバンドは分かりませんけど、ゲスの極み乙女。は分かります。中でもどんな曲が好きですか？\nB:今は、人生の針というのが好きです。\nA:人生の針ですか。私は知らない曲なので今度聞いてみます。\nB:ぜひ！ビデオもすごいこってるんですよ。ああいうところが好きですね。\nA:MVが凝ってるバンドって何か良いですよね。今度見てみます。\nB:ぜひぜひ！前でダンサーが寸劇をするんですが、逆再生で、メンバーはその後ろで普通に演奏してるんです。川谷氏は死体の役でも出演してますが。\nA:かなり凝った感じですね。川谷さんが死体役っていうのも興味深いです。\nB:でもそれはちょこっとだけですけど。トークではダンサーの人に思いきり蹴られて痛かったと言ってました。\nA:ダンサーに思い切り蹴られたんですか？それはお気の毒ですけど、ますます見てみたくなりました。\nB:監督が思い切りやってくださいと言ったら本気で蹴られたそうです。\nA:本気で蹴られたなんて相当痛かったでしょうね。でもきっといいMVになってるんだと思います。\nB:蹴るといっても座っている死体を押し入れに入れるためにやっているんですが。\nA:死体の状態で蹴られてるんですね。今は全然想像できてませんけど、見るのが楽しみです。\nB:考えようによっては怖いビデオです。可愛らしい女の子が毒入りのお菓子を楽しそうに作って。\nA:それは怖いですね。怖そうですけど歌も絶対聞いてみたいので見てみます。\nB:はい、後はビデオでお楽しみください。ということで10メッセージになりました。今日はどうもありがとうございました。\nA:こちらこそありがとうございました。\n"
    # dialogue3 = "A:今日はこちらは晴天です。そちらはいかがでしょうか。\nB:こちらは午前中晴れていましたが、午後からは曇りになりました。\nA:そうなんですね。寒い日が続きます。早く暖かくなってもらいたいものです。\nB:そうですね。こちらは最近少し雪も降りました。\nA:そうなんですね。今年は見れていないです。結構積もっているのですか。\nB:少しパラパラと降るくらいなので積もってはいなかったです。ただ地面が凍っていて滑りかけました。\nA:それは危なかったですね。最近家に居る事多いのでランニング始めようかと思っています。なにかスポーツやられていますか。\nB:このご時世ですし、ジムに行きにくいですよね。私は腹筋ローラーを買って筋トレしたり、ヨガをしています。\nA:なるほど、筋トレ・ヨガやりたくなりました。外まだ寒しい。\nB:ヨガもじんわり体が温まるのでおすすめですよ。ランニングも外に出た直後が寒いですね。\nA:そうなんですよ、子供が部屋でゲームばかりでなかなか出れないので。なにか動画見ながらですか？ヨガは。\nB:私もゲーム結構やってしまっています。\nB:動画ですね。YouTubeを見ながらヨガしています。\nA:YouTube様々なジャンルあるからいいですよね。他にどんなの見られますか？\nB:YouTubeも見出すとずっと見てしまいます。\nB:ASMRってご存知ですか？よく聞いています。\nA:存じ上げないです。どのようなジャンルなのですか？\nB:咀嚼音やスライムを練る音など、聞いていて気持ちがいい音を聞くジャンルです。\nB:包丁で石鹸をサクサク切る動画なんかも高音質で聞くとスカッとします。\nA:なんかTVで見たこと有るかもです。ずーと切ったり練ったりですよね、たしか\nB:そうですね。わりと好き嫌いわかれるジャンルかもしれませんが、私はついつい聞いてしまいますね。\nA:ありがとうございます。このあと早速見てみますね。\nB:ぜひ見てみてください。\nB:おもしろいと思います。\n"
    # summary = llm.generate_summary(dialogue=dialogue3, user_name="A")

# 観光地推薦文生成の検証
    llm = Llama_Swallow()
    rec_info =  "F1日本グランプリや鈴鹿8耐など国際的なレースも行われる鈴鹿サーキット。レースで実際に使用されている国際レーシングコースを走ることのできる「サーキットチャレンジャー」や、小さな子供がひとりで運転できる多彩なアトラクションが揃い、家族で楽しむことができる。プール(夏期限定)や天然温泉施設、地元の有機野菜を使用したレストランなどの施設も充実。"
    rec_sentence = llm.generate_rec_sentence_with_no_dialogue(rec_info=rec_info)
    print(rec_sentence)
    
   