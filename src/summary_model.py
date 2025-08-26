# -*- coding: utf-8 -*-
import logging
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re

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
    
    # def __generate_texts(self, prompt, num_sentence = 10):
    #     """
    #     Generates text based on the input prompt using the loaded language model.

    #     Args:
    #         prompt (str): The input text prompt to guide the text generation.

    #     Returns:
    #         str: Generated text based on the input prompt.
    #     """
    #     # Tokenize the input prompt
    #     inputs = self.tokenizer(prompt, return_tensors="pt")
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     input_ids_len = inputs["input_ids"].shape[1]
        
    #     # Perform inference to generate text
    #     with torch.no_grad():
    #             outputs = self.model.generate(
    #                 max_new_tokens=500,
    #                 inputs=inputs["input_ids"],
    #                 num_return_sequences=num_sentence,
    #                 no_repeat_ngram_size=2,
    #                 temperature=0.2,
    #                 top_k=0,
    #                 top_p=0.9,
    #                 early_stopping=True,
    #                 attention_mask=inputs["attention_mask"],
    #                 pad_token_id=self.tokenizer.pad_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id,
    #                 bos_token_id=self.tokenizer.bos_token_id,
    #                 num_beams=num_sentence * 2,
    #                 do_sample=True
    #             )
        
    #     generated_texts = []
    #     for output in outputs:
    #         generated_texts.append(self.tokenizer.decode(output[input_ids_len:], skip_special_tokens=True).replace(" ", "").replace("\n", "").replace(" ", "").replace("\n", ""))
    #     return generated_texts
    
    def __split_string_by_newline_count(self, dialogue, n):
        """
        Splits a string by a specified number of newline occurrences.

        Args:
            dialogue (str): The input string to be split based on newline occurrences.
            n (int): The number of newlines after which to split the text.

        Returns:
            list of str: List of text segments, each separated by `n` newlines.
        """
        sections = []
        start = 0
        newline_count = 0

        for i, char in enumerate(dialogue):
            if char == '\n':
                newline_count += 1
            # Split when the specified newline count is reached
            if newline_count == n:
                sections.append(dialogue[start:i + 1])
                start = i + 1
                newline_count = 0

        # Append remaining text if any
        if start < len(dialogue):
            sections.append(dialogue[start:])
        
        return sections
    
    
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

    def generate_summary(self, dialogue, temperature = 0.2):
        """
        Generates a summary for a given dialogue by processing the text with the model.

        Args:
            dialogue (str): The dialogue text to summarize.

        Returns:
            str: Generated summary of the input dialogue.
        """
        
        dialogue_list = self.__split_string_by_newline_count(dialogue, n=80)
        # summary_list = []
        for _ in range(10):
            summary_list = []
            for short_dialogue in dialogue_list:
                    # Tokenize the input dialogue
                prompt = f"""
【タスク】
AとBの対話履歴からBの観光地に対する趣味嗜好や習慣，プロフィールについてまとめてください．
以下の事項を必ず守ってください
- Bが観光地に求める条件を抽出してください
- 出力は改行などせずに必ず一文でお願いします
- 出力は構造化しないでください
- 対話で言及されていない内容を含めないでください
- URLなどは出力しないでください

--Example 1--
【対話履歴】
A:さっそくですが、なにか趣味などありますか？
B:インドアなので、家でゲームばかりです。どんな趣味をお持ちですか。
A:私もインドア派なので、趣味は映画鑑賞やお菓子作りなど家でできることばかりです。どんなゲームが好きですか？
B:難しいアクションとかやります。お菓子作りいいですね。あとは散歩は割としますね。
A:ひょっとして犬を飼っているのですか？
B:犬飼ってみたいです、願望はあるのですが。散歩といっても沢山歩くのは嫌で、近所を一周して終わりとかです。
A:いいですね。私も散歩を趣味にしたいです。でも以前犬を飼っていたときはよく散歩にいったのですが、一人だと散歩をどう愉しめば良いのかわかりません。
B:確かに散歩は目標とかは立てにくいですしね。なんかずっと家にいると気分が滅入っちゃうので、外の空気を吸いたい、て感じです。
A:知らない土地ならまだワクワクするのもわかるのですが、見慣れた近所を歩く楽しみのコツてなにかありますか？
B:初めていく土地はたしかに見るだけで楽しいですね。地味ですが、姿勢を意識して歩いて、健康改善の目論見もあります。
A:健康のためということを意識するといいのかもしれませんね。走ることは苦手なので、せめて散歩でもして体を動かしたいです。
B:近くに桜の木があるので、春はそこを歩くのが楽しいです。
A:目的地があったり、途中の景観がいいとやりがいがありますね。最近はGPSと連動できる散歩用のアプリなんかもありますよね。
B:目的地あるといいですよね。散歩用のアプリ、まさに使ってます。ちなみにどんな景色が好きですか。
A:自然の景色が好きですが、特に好きなのは砂漠の景色です。なかなか日本では難しいですが。どんな景色が好きですか？
B:砂漠の景観、確かに日常から遠い世界ですね。桜も好きですが、秋の紅葉とか、美しいなと感じます。
A:どちらも日本ならではの風物詩ですね。山歩きをしたくなりますね。
B:登山あこがれます。体力も道具もないけど。誰かそういうの好きな人に連れてってもらいたいです。
A:たしかに登山は危険な面もありますからね。一人で登山をしていて道に迷ったかと不安になったことがありますが、本当に怖かったです。
B:ご自身の体験なんですね。恐ろしい。でもいつかは一人登山してみたいです。
A:自然は神秘的だから惹かれますね。楽しいお話ありがとうございました。

【要約】
インドア派で、ゲームが趣味です。また、散歩も好きで姿勢を意識しながら歩くことや健康改善を目指しています。桜の木のある場所で散歩するのが楽しいと話しており、自然の景色や秋の紅葉が好きです。また、登山に憧れており、一人での登山経験に興味があります。


--Example2---
A: 本日はご利用いただきましてありがとうございます。お客様、今日はご旅行のご相談でよろしいでしょうか。
B: そうですね。はい、よろしくお願いします。
A: よろしくお願いします。お客様どちらへご旅行になりますでしょうか？
B: はい。 そうですね、なんとなくしか決めてないんですけど、えっと夫婦、50代の夫婦二人で，秋なので、なんとなくなんですけど、箱根方面に行きたいかなーって思うんですけど。
A: ええ、ええ。 とても素敵な旅行になりそうですよね。
B: はは、ありがとうございます。
A: ええ、ではそれではですね、お客様のご要望がご夫婦お二人で、<>の箱根へ行かれるということでよろしいでしょうか？
B: はい、ええ、はい、 あっ、そうです。はい。
A: かしこまりました。それではまず箱根のほうでお調べ致しますので少々お待ちくださいませ。
B: はい、あっおねがいします。
A: はい。 それではお客様、箱根なんですけれども、どういったことなさりたいかとか、あるいは何か具体的にもうここにいきたいとかございますか？
B: はい。
A: あるいはぼんやりとどういったことしたいなとかあればこちらの方でお調べ致します。
B: あっ、はい。えっとそうですね、何か所か<>とこがあるんですけど
A: ええ。
B: あのーちょっとあの名称とかはわからないんですけど
A: ええ。
B: 確か何かオルゴール、あっオルゴールの美術館的なものがあるって<>で、
A: ええ、あっええ。あっはい。
B: そういうなんか、ぶん、文化的な行動をしたいと思って、はい。
A: ええ、あっ、なるほど。 じゃあお客様、まずそのオルゴールの美術館をお調べしてみましょうか？
B: はい、あっはい、おねがいします。
A: 少々お待ちくださいませ。
B: はい。
A: はい、お待たせしております。 オルゴールの美術館というものが、こちらでご案内できる場所がですね、
B: はい、はい。
A: 少々お待ちくださいね。こちら、アンノイエというところがあるんですけれども、こちらお客様ご存知でしょうか？
B: アンノイエですか？
A: ええ。
B: えっとアンというのはどんな字を書きますか？
A: こちらカタカナでアンと書きます。
B: あーなるほど。 アンの、うん、うん、はい、あっそれは知らなかったです。はい。
A: アンノイエ、<>",

【要約】
50代の夫婦二人で旅行を計画しており，箱根方面に旅行をしたいと考えている。オルゴールの美術館に行きたいと考えている。

--Let's begin!--
【対話履歴】
{short_dialogue}

【要約】
                
"""
                generated_summary = self.__generate_text(prompt, temperature)
                summary_list.append(generated_summary)
                
            all_short_dialogue_summary = "\n".join(summary_list).strip("\n")

            prompt=f"""
以下の内容をもとにBの観光地に対する趣味・経験を要約してください。
なるべく多くの情報を含む要約文を生成してください。

【要約元文章】
{all_short_dialogue_summary}

【要約文章】

"""
            last_summary = self.__generate_text(prompt, temperature=temperature)
            # print(len(last_summary))
            if len(last_summary)>=30 and self.calculate_japanese_ratio(last_summary)>0.9:
                break
    
        return last_summary


    def generate_summaries(self,dialogue, num_sentence=10):
        dialogue_list = self.__split_string_by_newline_count(dialogue, n=80)
        for i in range(5):
            summary_list = []
            for short_dialogue in dialogue_list:
                        # Tokenize the input dialogue
                prompt = f"""
【タスク】
AとBの対話履歴からBの観光地に対する趣味嗜好や習慣，プロフィールについてまとめてください．
- Bが観光地に求める条件を抽出してください
- 出力は改行などせずに必ず一文でお願いします
- 出力は構造化しないでください
- 対話で言及されていない内容を含めないでください
- URLなどは出力しないでください

--Example 1--
【対話履歴】
A:さっそくですが、なにか趣味などありますか？
B:インドアなので、家でゲームばかりです。どんな趣味をお持ちですか。
A:私もインドア派なので、趣味は映画鑑賞やお菓子作りなど家でできることばかりです。どんなゲームが好きですか？
B:難しいアクションとかやります。お菓子作りいいですね。あとは散歩は割としますね。
A:ひょっとして犬を飼っているのですか？
B:犬飼ってみたいです、願望はあるのですが。散歩といっても沢山歩くのは嫌で、近所を一周して終わりとかです。
A:いいですね。私も散歩を趣味にしたいです。でも以前犬を飼っていたときはよく散歩にいったのですが、一人だと散歩をどう愉しめば良いのかわかりません。
B:確かに散歩は目標とかは立てにくいですしね。なんかずっと家にいると気分が滅入っちゃうので、外の空気を吸いたい、て感じです。
A:知らない土地ならまだワクワクするのもわかるのですが、見慣れた近所を歩く楽しみのコツてなにかありますか？
B:初めていく土地はたしかに見るだけで楽しいですね。地味ですが、姿勢を意識して歩いて、健康改善の目論見もあります。
A:健康のためということを意識するといいのかもしれませんね。走ることは苦手なので、せめて散歩でもして体を動かしたいです。
B:近くに桜の木があるので、春はそこを歩くのが楽しいです。
A:目的地があったり、途中の景観がいいとやりがいがありますね。最近はGPSと連動できる散歩用のアプリなんかもありますよね。
B:目的地あるといいですよね。散歩用のアプリ、まさに使ってます。ちなみにどんな景色が好きですか。
A:自然の景色が好きですが、特に好きなのは砂漠の景色です。なかなか日本では難しいですが。どんな景色が好きですか？
B:砂漠の景観、確かに日常から遠い世界ですね。桜も好きですが、秋の紅葉とか、美しいなと感じます。
A:どちらも日本ならではの風物詩ですね。山歩きをしたくなりますね。
B:登山あこがれます。体力も道具もないけど。誰かそういうの好きな人に連れてってもらいたいです。
A:たしかに登山は危険な面もありますからね。一人で登山をしていて道に迷ったかと不安になったことがありますが、本当に怖かったです。
B:ご自身の体験なんですね。恐ろしい。でもいつかは一人登山してみたいです。
A:自然は神秘的だから惹かれますね。楽しいお話ありがとうございました。

【要約】
インドア派で、ゲームが趣味です。また、散歩も好きで姿勢を意識しながら歩くことや健康改善を目指しています。桜の木のある場所で散歩するのが楽しいと話しており、自然の景色や秋の紅葉が好きです。また、登山に憧れており、一人での登山経験に興味があります。


--Example2---
A: 本日はご利用いただきましてありがとうございます。お客様、今日はご旅行のご相談でよろしいでしょうか。
B: そうですね。はい、よろしくお願いします。
A: よろしくお願いします。お客様どちらへご旅行になりますでしょうか？
B: はい。 そうですね、なんとなくしか決めてないんですけど、えっと夫婦、50代の夫婦二人で，秋なので、なんとなくなんですけど、箱根方面に行きたいかなーって思うんですけど。
A: ええ、ええ。 とても素敵な旅行になりそうですよね。
B: はは、ありがとうございます。
A: ええ、ではそれではですね、お客様のご要望がご夫婦お二人で、<>の箱根へ行かれるということでよろしいでしょうか？
B: はい、ええ、はい、 あっ、そうです。はい。
A: かしこまりました。それではまず箱根のほうでお調べ致しますので少々お待ちくださいませ。
B: はい、あっおねがいします。
A: はい。 それではお客様、箱根なんですけれども、どういったことなさりたいかとか、あるいは何か具体的にもうここにいきたいとかございますか？
B: はい。
A: あるいはぼんやりとどういったことしたいなとかあればこちらの方でお調べ致します。
B: あっ、はい。えっとそうですね、何か所か<>とこがあるんですけど
A: ええ。
B: あのーちょっとあの名称とかはわからないんですけど
A: ええ。
B: 確か何かオルゴール、あっオルゴールの美術館的なものがあるって<>で、
A: ええ、あっええ。あっはい。
B: そういうなんか、ぶん、文化的な行動をしたいと思って、はい。
A: ええ、あっ、なるほど。 じゃあお客様、まずそのオルゴールの美術館をお調べしてみましょうか？
B: はい、あっはい、おねがいします。
A: 少々お待ちくださいませ。
B: はい。
A: はい、お待たせしております。 オルゴールの美術館というものが、こちらでご案内できる場所がですね、
B: はい、はい。
A: 少々お待ちくださいね。こちら、アンノイエというところがあるんですけれども、こちらお客様ご存知でしょうか？
B: アンノイエですか？
A: ええ。
B: えっとアンというのはどんな字を書きますか？
A: こちらカタカナでアンと書きます。
B: あーなるほど。 アンの、うん、うん、はい、あっそれは知らなかったです。はい。
A: アンノイエ、<>",

【要約】
夫婦二人で旅行を計画しており，箱根方面に旅行をしたいと考えている。オルゴールの美術館に行きたいと考えている。

--Let's begin!--
【対話履歴】
{short_dialogue}

【要約】
                
"""
                generated_summary = self.__generate_text(prompt, temperature=1.0)
                summary_list.append(generated_summary)
            all_short_dialogue_summary = "\n".join(summary_list).strip("\n")
            if len(all_short_dialogue_summary)>10:
                break

        generated_summaries = []
        for i in range(num_sentence):
            generated_summaries.append(self.__generate_text(all_short_dialogue_summary, temperature=0.7))
        return all_short_dialogue_summary, generated_summaries
    
    def generate_rec_sentence_with_no_dialogue(self, rec_info: str, temperature=0.2):
        prompt = f"""
# 観光地情報をもとに観光地推薦文を生成してください。

==Example1==
【観光地情報】
札幌市内南東部に位置する全天候型屋内ドーム。北海道コンサドーレ札幌、北海道日本ハムファイターズのホームスタジアムで、サッカーや野球の試合のほか、スポーツ・コンサートなど各種イベントも行われる。買い物や食事が楽しめるショップ、見晴らしのいい展望台あり。イベントのない時には、札幌ドームの裏側を見学するドームツアーも開催。所要90分以上～ ベビーおすすめ キッズおすすめ 女子おすすめ 冬におすすめ 雨でもOK

【観光地推薦文】
全天候型屋内ドームであり、北海道コンサドーレ札幌、北海道日本ハムファイターズのホームスタジアム。サッカーや野球の試合のほか、スポーツ・コンサートなど各種イベントも行われ他、買い物や食事も楽しむことができる。見晴らしの良い展望台があり，良い景色を見ることができる。お子様連れの方や雨の日でも観光したい方におすすめの観光地です。

==Let's begin!==
【観光地情報】
{rec_info}

【観光地推薦文】
"""
        rec_sentence = self.__generate_text(prompt, temperature=temperature)
        return rec_sentence
    
    def generate_rec_sentence_with_dialogue(self, rec_info: str, dialogue_summary: str):
        prompt = f"以下の観光地情報をもとに観光地推薦文を生成してください。ユーザの特徴文を考慮して作成してください。\n【観光地情報】\n{rec_info}\n\n【ユーザ情報】{dialogue_summary}"
        rec_sentence = self.__generate_text(prompt)
        return rec_sentence
    


if __name__ == "__main__":
    # 対話要約分生成の検証
    dialogue ="A: 本日はご利用いただきましてありがとうございます。\nB: よろしくお願いしまーす。\nA: よろしくお願い致します。 お客様、今日はご旅行のご相談でよろしいでしょうか。\nB: はい。\nA: はい、ありがとうございます、どちらへのご旅行になりますでしょうか。\nB: えっとですねー、ちょっと週末に沖縄に行くことになってー。\nA: ええ。 左様でございますが、では沖縄へのご旅行でございますねー。\nB: ええー。 そうですー。\nA: ありがとうございます、いつごろ、もう週末というのは、今度の週末とかでよろしいですか。\nB: えーとですね、再来週の週末なんですけどー。\nA: では、えー、再来週、まあ冬の時期にあったご旅行先ということでよろしいでしょうか。\nB: はい、はい。\nA: かしこまりました、では何名様でのご旅行になりますか。\nB: えっと、一人なんですけどー。\nA: お1人様でございますね。\nB: はい。\nA: かしこまりました、ではえー、本日承りますのがー、こちら沖縄へのご旅行で、えー、時期が冬、一名様ということでお間違えないでしょうか。\nB: はい。 はい。 はい、そうです。\nA: ありがとうございます、ではまずですね、沖縄に行ってメインでどういうことをしたいか、決めていきませんか？\nB: あ、はい、お願いしまーす。\nA: はい、ではですね、お客様、沖縄に行って何をしたいか何かご希望ございましたら、アイデアをいただけますでしょうか。\nB: そうですね、まあ沖縄と言えば、海なんで、ビーチにちょっと水遊びじゃないですけど、ゆっくりしたいなあとか、リラックスですね、はい。 ええ、ええー。 ええ。 ええー。 ええ<>でございますね。\nA: そうですね、あの、沖縄、冬でもまあもちろん本州よりあったかいですし、現地に行くとお楽しみいただけるかもしれませんよね。\nB: はい。 はい。\nA: はい、ではビーチでお調べしてもよろしいでしょうか？\nB: あ、お願いしまーす。\nA: かしこまりました、ではお客様、あのー、沖縄のなかでー、例えば7のような南部と、あと北部、どちらがよろしいでしょうか。\nB: あー、ちょっと沖縄よく分からなくてー、あ、でも島行ってみたいです。\nA: あるいは島<>。 ええー、ではですねえ、お客様、例えば、沖縄の本島のすぐ横にあるような島、橋でつながっているような島、そういったものもございますし、あとですねー、あのー海沿いにビーチパラソルがたーくさん並んでいるようなビーチもございますしー、<>もっと手軽にですねー、ものあの那覇の市内ですね、なんかあのー、ビーチの向こう側に、あのー道路のはしが見えるような<>、ちょっとあのー、ビーチの向こうに道路のはしが見えるのは、なんか<>\nB: あー、うーん、そういうのも。 うーん。 ええー。 はい。 あー。 <>ちょっとしらけちゃうかな、うーん。\nA: そうですよね、なんか、都会感が強すぎますしー<>。\nB: うーん。 できれば離島に行ってみたいんですけどー。\nA: ええー、左様でございますか、では、那覇からはな、沖縄の本島ですね、から離れた離島がよろしいでしょうか、それとも本島からすぐに行ける離島がよろしいでしょうか。\nB: はい。 はい。 あー、近いほうがいいですねー、あんまり移動の時間はとりたくないんでー。\nA: ええ。 ええ、それではですねー、えー、こちら奥武島という場所がございまして、奥武島、こちらですね、はい、漢字で書くと、えー、手前奥の奥という字に、武士の武、あ、すいません、武士の武ですね、えーこれに、奥に武と書いておうと呼んで、さらに島アイランドですねー、それで奥武島という場所がございます。\nB: はい。 奥武島、はい。 はい。 あー、は、はい。 あー。 はい。 はい。\nA: こちらなんですけどー、沖縄の本島からですねー、橋でつながっておりますので、はい、なんとあの車とかで徒歩でも行き来ができるような非常に、ええ、便利な場所にはあるんですけれども、なおかつ島の雰囲気はのどかということです。 あー。 あー、そうですか。 はい。 はい。 あー、へー、面白そうですね。\nB: ええ、<>、ええー、良いかなあというふうに思いまして、でそのー、島に入って、その橋を渡るとすぐにビーチがあるようなんですね。 うん。 あー、はい。\nA: アクセスとしても非常に便利ですしー、こちらですねえ、えー、那覇のバスターミナルから40分となっておりますのでー、ええー、アクセス良く、えー、例えばですねこのビーチを楽しんだ後にほかのものを見るとかー、そういった点でもこちらの立地は非常に良いのではないかなと思ったんですけど、ええー、いかがでしょうか。\nB: あー。 はい。 はい。 あーそうですかー、ええー。 あーちょっとそこ候補にしたいですね。\nA: かしこまりました、ありがとうございます、では奥武島を一つ候補に取り上げますねー。 はい。 はい、はい。\nB: はい、ありがとうございます、ではお客様ビーチ以外にも沖縄たくさんお楽しみいただけるところございますが、どうでしょうか。\nA: あー。 もっとたくさんもうビーチづくしで、次もビーチ、前もビーチみたいな感じで行くのとー、それかほか例えばお買い物をするとかー、何か、まあもちろんお食事は別途お考えいただくのが良いんですけれども、まあお買い物系とかー、あるいはどうでしょう、もっとビーチで行くか、なんか別のアトラクションがいいか。\nB: <>はー。 ええー。 うーん、ええ。 あー。 ちょっと話が飛んでしまうんですけどー、離島でなんか牛がひっぱる車に乗るのってあるじゃないですか<>あれちょっと一回乗ってみたいんですよねー。\nA: ええー。 ええ。 ええ、ええー、ありますね、あ、はいはい、ではそちらお調べしてみますね、少々お待ちくださいませ。\nB: はい。\nA: お客様、お待たせいたしました。\nB: はい。\nA: はい、え、こちらお楽しみいただけるのが、ビオスの丘というところ。\nB: ビオス、ビオスの丘\nA: はい、ビオスとカタカナで書きます。\nB: はい。\nA: ビオスの丘と言いましてー、こちらにですねー、水牛がひっぱる車に乗って道を歩く体験ができたりー、あとですね、それ以外にもここですとカヌーを楽しんだりー、であと、亜熱帯の林のなかにある川をですねー、船に乗って下るとか、そういったアトラクションがございましてー、ええ。\nB: はい。 ええ。 うーん。 あー、はい。 あー。 えー。 ええー、まあそれは本島ですか？\nA: ええ。 はい、こちら本島でございます、ですので。\nB: あ、うーん、どの辺なんでしょうねえ。\nA: お客様<>。 うるま市というところにございましてー、ええ、ですのでお客様がおっしゃられていた離島のものとはちょっと違うんです、離島のものの方がよろしいでしょうか、離島ですと、こちらですね、えー、石垣島になりますね。\nB: はい。 うーん。 うーん。 ええー。\nA: 石垣島の方にも、こちら水牛が車をひっぱってー、海の中の、なんていうの海岸沿いですか。\nB: うーん。 あー、それがいいですねー、うーん。\nA: あ、こちらのイメージでございますか。\nB: あ、左様でございますか、大変失礼しました。\nA: うーん。 ではですね、こちら平田観光というところがやっておりましてー。\nB: はい。 はい。\nA: ええ、えー例えばまあ、ここの場合もですね、あのー川のボート遊覧があったりー、あと先ほど申し上げました水牛の車で海沿いを歩くような観光があったりー、はい、あとセグウェイに乗って、セグウェイってお客様ご存知ですか。\nB: はい。 はい。 セグウェイ、いやなんでしょうねえ。\nA: なんていうんでしょう、なんかあの車みたいなやつの上に立って乗るんですけれどもー、自転車みたいにこぐんではなくて、まあ勝手に動いてくれる二輪車のようなものでしょうか。\nB: ええー。 うーん。 あー、ちょっとスポーツ的なものですかね、そしたら。\nA: そうですね、ちょっとまあ体力を使うような<>それにお客様ちょっと冬ですと、このセグウェイ、外は寒いかもしれませんね。\nB: あー。 あっは、あーそうですね、あんまり汗かくことはしたくないんでー。\nA: ええー、ですので<>。 ええー、そうですよね、ではもしよろしければ、こちらの水牛の車に乗る観光がちょうどよいかなあと思ったんですけれども<>。\nB: はい。 あ、はい、それがいいです。\nA: ではこちら平田観光と言いますので、こちらをリストに加えておきますね。\nB: はい。 あ、分かりましたー。\nA: では、お客様せっかくですので、沖縄で何かお食事なさいませんか。\nB: あー、なんでしょうね、沖縄、おすすめとかあったら、教えていただきたいんですけど。\nA: ええー、沖縄の。 はい、ではですねー、沖縄の郷土料理でお調べしてみましょうか。\nB: はい。\nA: はい。 例えばですねー、これは郷土料理といってよいのか分からないですけれどもー、お客様、アグー豚ってご存知でしょうか？\nB: はい。 あー東京でも売ってますよね、うーん。\nA: ええー、あ、そうですよね、お客様そうなりますと、アグー豚のしゃぶしゃぶ食べても、なんか沖縄って感じしませんかね？\nB: はい。 あー、あっは、あ、でも好きなんでー、OKです。\nA: ええー、あそうですか、えー、私共の方に映っている写真を拝見しますとー、アグー豚のしゃぶしゃぶでしたりー、あとーちょとレア感が残ったー、えー、アグー豚のとんかつ<>そういう、お客様がよろしければ、もちろんこちらおすすめなんですけどもー、ちょっとこれ東京でも食べられるよって、あのつっこまれるときついところがございますので、もしよろしければ、ほかのところもお調べしましてー、比較していただくのが一番いいかなと思います。\nB: うーん。 あー、うーん。 はい。 <>うーん。 あ、そうですねー。\nA: ではですね、ほかのジャンルもお調べしてよろしいですか。\nB: はい、お願いします。\nA: はい、じゃあ少々お待ちください。 うーん、沖縄ちょっとお魚って感じでもないのかなあっていう気はするんですけれどもー、ちょっとそうですね、ちょっと郷土料理<>うーん。\nB: そうですねー。 なんだろう、ゴーヤとかそういうもの<>うーん。\nA: あ、ゴーヤでございますか、お客様ゴーヤはお好きですか。\nB: 好きですー。\nA: あ、ではですねー、例えばですねー、えーとゴーヤのサラダがあるようなお店<>。\nB: あー。\nA: ただ、お客様、やっぱゴーヤはですね、夏が旬のようですのでー。\nB: あー、そうでしたねー。\nA: そのかわりにですね、お客様、冬ですと島にんじんってのがあるそうです。\nB: あー、島にんじん、ふーん。\nA: ええ、ちょっと私のほうでは存じ上げないですけれどもー、こちら書いている説明では島にんじんなどの旬の島野菜が食べられるお店というのがございましてー。\nB: うーん。 ええー。\nA: で、なんかゴーヤチャンプルーとかまあ、おそらく冬でもゴーヤ自体は一応あるということなんでしょうけれども、まあ名前がですね、沖縄家庭料理あかさたな市場店となっておりますねー、なので沖縄の家庭料理全般が食べられてー、まあゴーヤは旬ではないけれどもー、えー、ゴーヤチャンプルーとかはあったり、あとまあ冬は島にんじん、そういった感じですかねー、ちょっとおきゃ、ええ。\nB: うーん。 あー。 うーん。 うーん。 あー。 まあ、一通りそろうってことですよね。\nA: そうですね、まああの沖縄行ってきましたーっていうこう話ができるような場所ではあるのかなあと思います。\nB: うん。 うーん、あー。 あー。\nA: そのーほかにもですねー、なんか沖縄料理ということでー、たまごのなんていうんでしょう、スクランブルエッグの上に、えー、スパムっていうんですかね、それがのって、そうですそうです、ただちょっとなんか、沖縄まで行ってこれ食べる<>感じもしますしー、ええ、ちょっと待ってくださいねー、あのー、私共のほうで調べてみますので、お待ちくださいませ。\nB: あー、缶詰の肉ですよね。 <>なかなか難しいですねー、うーん。 はい。\nA: あ、お客様、麺類はお好きですか？\nB: あ、好きですー。\nA: あ、そうですか、あのー、こちらですねー、沖縄のソーキそばがございますねー。\nB: あー、うーん、まあ沖縄料理は全般的に好きなんで何が出てきてもいいです。\nA: ええー、ええー なるほど、そうですねー、なんかそのお店のご提供の仕方としてはー、ソーキそばのところはソーキそばばっかりが多くて、まあ<>、そうですね、そうでしたらやっぱソーキそばも食べれるしー、えー、ほかのなんか沖縄料理のようなものも食べれるお店のほうがいいですかね。\nB: うーん。 あー、はい。 うーん、できれば、そのほうがいいですね、いろいろつまめるみたいな、うーん。\nA: ええ、そうですよね、ええー、あ、なるほど、そうしますと、そういったぴったりのお店をお調べしますと、あ、お客様、まあ先ほどあのアグー豚、あのお好きということだったんですけれどもー、アグー豚とかー、あともとぶ牛、牛ですね。\nB: はい。 ええー。 もとぶ牛、はい。\nA: ええ、もとぶっていうのは、おそらく沖縄の北部の地名だと思うんですけれどもー、ええ、こちらあのー、もとぶの牛、こちらA5、A4ランクとなっておりましてー、あとやんばる若鶏<>、ええ、そういったももあるしー、ゴーヤチャンプルもあるしー、こちらですねー私共の方の画面でうつっている写真では、島どうふってお客様ご存知です？\nB: はい。 あー。 あー、いろいろあるんですねー。 うーん。 うーん。 はい。 あ、知ってます、うちの地元でも売ってま<>。\nA: ええー、あ、そうなんですか、お客様お好きですか、島豆腐。\nB: あ、好きですー。\nA: あ、良かった、そうしますと、ゴーヤチャンプルも食べれるしー、島豆腐も食べれるしー、もとぶ牛も食べれるしー、アグー豚も食べれるしー、やんばる若鶏も食べれ、たぶん全部は食べれないですけれども、ええ、そうですよねー<>。\nB: はい。 うーん。 はーい。 あっはっは、ちょっと食べきれるか分からないけど、でもこのお店いいですね。\nA: そうですね、お客様、どんな気分で行っても決して裏切られることはないお店かと思います。\nB: あー、イチオシですか？\nA: んーもっといいところあるかもしれませんけどー<>そうですね、あ。\nB: <>とりあえずその辺で、うん。"
    llm = Llama_Swallow(model_name="dpo-summary-results-new")
    summaries = llm.generate_summary(dialogue=dialogue, temperature=0.2)
    print("Generated Text:", summaries)
    print(type(summaries))

    # 観光地推薦文の検証

    # llm = Llama_Swallow()
    # rec_info =  "北大植物園の園内にある、初代園長、宮部金吾博士に関する記念館。北海道や千島、樺太を踏破し、北方に育つ植物の研究を重ねた功績が遺品とともに展示されている。明治34年(1901)に建てられ、動植物学講堂や庁舎として使用されてきた建物そのものも歴史を静かに語る。所要30～60分くらい 女子おすすめ 雨でもOK"
    # for i in range(15):
    #     summary = llm.generate_rec_sentence_with_no_dialogue(rec_info=rec_info)
    #     print("Generated Text:", summary)
    