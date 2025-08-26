# -*- coding: utf-8 -*-
import logging
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Hugging Faceキャッシュディレクトリの設定
os.environ['HF_HOME'] = os.path.join(BASE_DIR, "../../.venv/cache")  # カレントディレクトリ基準の相対パス

logging.getLogger("transformers").setLevel(logging.ERROR)

class Llama_Swallow:
    """
    A class to handle the loading and inference of the Llama-3.1-Swallow-8B model for text generation and summarization tasks.
    """

    def __init__(self, model_name: str = "./dpo-recommendation-results"):
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

    def generate_rec_sentence_with_no_dialogue(self, rec_info: str, temperature=0.2):
        prompt = f"""
あなたはプロの観光地推薦者です。
観光地情報をもとに観光地推薦文を生成してください。観光地推薦文は一文で出力をしてください。

--Example1--
【観光地情報】
札幌市内南東部に位置する全天候型屋内ドーム。北海道コンサドーレ札幌、北海道日本ハムファイターズのホームスタジアムで、サッカーや野球の試合のほか、スポーツ・コンサートなど各種イベントも行われる。買い物や食事が楽しめるショップ、見晴らしのいい展望台あり。イベントのない時には、札幌ドームの裏側を見学するドームツアーも開催。所要90分以上～ ベビーおすすめ キッズおすすめ 女子おすすめ 冬におすすめ 雨でもOK

【観光地推薦文】
全天候型屋内ドームであり、北海道コンサドーレ札幌、北海道日本ハムファイターズのホームスタジアム。サッカーや野球の試合のほか、スポーツ・コンサートなど各種イベントも行われ他、買い物や食事も楽しむことができる。見晴らしの良い展望台があり，良い景色を見ることができる。お子様連れの方や雨の日でも観光したい方におすすめの観光地です。

--Let's begin!--
【観光地情報】
{rec_info}

【観光地推薦文】
"""
        rec_sentence = self.__generate_text(prompt, temperature=temperature)
        return rec_sentence


if __name__ == "__main__":
    # 対話要約分生成の検証
    # dialogue ="A: 本日はご利用いただきましてありがとうございます。お客様、今日はご旅行のご相談でよろしいでしょうか。\nB: そうですね。はい、よろしくお願いします。\nA: よろしくお願いします。お客様どちらへご旅行になりますでしょうか？\nB: はい。 そうですね、なんとなくしか決めてないんですけど、えっと夫婦、50代の夫婦二人で\nA: ええ、ええ。\nB: んーなんかちょっと秋なので、なんとなくなんですけど、箱根方面に行きたいかなーって思うんですけど。\nA: ええ、ええ。 とても素敵な旅行になりそうですよね。\nB: はは、ありがとうございます。\nA: ええ、ではそれではですね、お客様のご要望がご夫婦お二人で、<>の箱根へ行かれるということでよろしいでしょうか？\nB: はい、ええ、はい、 あっ、そうです。はい。\nA: かしこまりました。それではまず箱根のほうでお調べ致しますので少々お待ちくださいませ。\nB: はい、あっおねがいします。\nA: はい。 それではお客様、箱根なんですけれども、どういったことなさりたいかとか、あるいは何か具体的にもうここにいきたいとかございますか？\nB: はい。\nA: あるいはぼんやりとどういったことしたいなとかあればこちらの方でお調べ致します。\nB: あっ、はい。えっとそうですね、何か所か<>とこがあるんですけど\nA: ええ。\nB: あのーちょっとあの名称とかはわからないんですけど\nA: ええ。\nB: 確か何かオルゴール、あっオルゴールの美術館的なものがあるって<>で、\nA: ええ、あっええ。あっはい。\nB: そういうなんか、ぶん、文化的な行動をしたいと思って、はい。\nA: ええ、あっ、なるほど。 じゃあお客様、まずそのオルゴールの美術館をお調べしてみましょうか？\nB: はい、あっはい、おねがいします。\nA: 少々お待ちくださいませ。\nB: はい。\nA: お客様\nB: はい。\nA: はい、お待たせしております。 オルゴールの美術館というものが、こちらでご案内できる場所がですね、\nB: はい、はい。\nA: 少々お待ちくださいね。こちら、アンノイエというところがあるんですけれども、こちらお客様ご存知でしょうか？\nB: アンノイエですか？\nA: ええ。\nB: えっとアンというのはどんな字を書きますか？\nA: こちらカタカナでアンと書きます。\nB: あーなるほど。 アンの、うん、うん、はい、あっそれは知らなかったです。はい。\nA: アンの家、<> さようでございますか。こちらですね、ガラス細工であったり、あるいは犬とか猫とか動物のような形をした、まぁガラス細工ですね。 それとか、オルゴール。そういったものが置かれております。\nB: あっはい。\nA: で、お客様、こちらはお客様がイメージされていたものを、に合いますでしょうか？\nB: そうですね。あの、プラス、プラスオンであのガラス細工というところがとても気に入りまして\nA: ええ。ええー。\nB: あと、えっと動物をモチーフしたものってところも、猫を飼ってるので、ちょっと行ってみて、なんか\nA: ええ。ええ、ええ。\nB: <>がもしあれば、<>かななんて、ちょっともうちょっと夢が膨らみました。ありがとう、はい。\nA: ええ、えっあっありがとうございます。 はい。ではこちらを行かれる観光地の一つに加え<>ようか。\nB: そうですね。はい。\nA: ええ、ありがとうございます。\nB: はい。じゃあ他にどういったところがお好みか、ご希望ございますか？ あっそうですね。あの、はいざっくりとでも、ええ、かまいませんので。 あの先ほど申し上げましたけども、ちょっと文化的な\nA: ええ、ええ。\nB: 線とかではなくて、文化的なちょっと、行動をしたいと思ってるので夫婦二人で\nA: ええ、ええ、ええ。\nB: オルゴールは今あったので、そうだなーえっとそうですね。\nA: ええ。\nB: あっ、ほんとですか。 何か、こう、んー なんかびじゅ、なんかこうキーワードで美術とか文化とかそういうものありますか？\nA: ええええ。 ええ、例えばですねじゃあお客様\nB: はい。\nA: 和風の歴史がよろしいか\nB: はい。\nA: あるいはですね、文学だとか、\nB: うんうん\nA: あるいはちょっとですね、科学的なんですけども、地球とか星に関する博物館\nB: はい、あーうん。 はい。\nA: それとか、あとですねちょっと箱根からずれるんですけども\nB: はい。\nA: かまぼこ博物館というものがございますね。\nB: ふふ、はい。\nA: で、お客様こちらの場所はですね、小田原になるんですけれども\nB: ええええ。はいはい。\nA: 同じエリアですので行くことには問題ござ<>ええ。\nB: そうですね。うん。\nA: あの、かまぼこちょっとですね、趣向が違っておりま<>ええ。\nB: そうですね。うんうんうん。\nA: お客様、他にもですね、<>、あとまぁ郷土資料館ですとか、まだまだですね文化施設は沢山御座いまして\nB: はい。\nA: 美術館もございますし\nB: はい。\nA: あと屋外展示場であったり。\nB: うんうんうんうん。\nA: あの秋ですのでちょうど気持ち良い季節かと思いますので\nB: そうですね、はい。\nA: 外歩き回ってその美術の展示をご覧いただくのもよいかと思います。\nB: はい、ああ、はい。はい。\nA: お客様、どういっ、あと彫刻の森美術館もいいかなぁというに思いますね。\nB: うーん、はい、はい。\nA: どうですかお客様どう<>何かイメージがわいてきましたか？\nB: そうです、あっ、そうですね。あっあのーえっと、まぁ最後にあのーおっしゃってくださった彫刻の森美術館なんですけれども\nA: ええ、ええ。\nB: メジャーなところではなくて、ひとつ前にご提案してくださった、かまぼこ博物館\nA: ええ、ええ、ええ。\nB: そういうちょっとこうB、B級といういい方はちょっとあれ\nA: ええええ。\nB: そういうなんか、ね、ネタになりそうな、<>とか、あとなんだろう\nA: ええ、ええええ。\nB: あの三つ、は、あのはじめにセンテンスがあって、えっと二つ目のあの文学とかそういう部分を特化したところ\nA: ええ。ええ、ええ。\nB: と、まぁ秋なので、夜空もいいのかなと思ったので、星とかですか？\nA: ええ、ええ。\nB: 地球とか、なんかそういう感じのところにもうちょっと絞って<>ますか？\nA: はい、ええええ。 かしこまりました。\nB: はい。\nA: ではですね、まずお客様が最初におっしゃられました、かまぼこの博物館\nB: はい。はい。そうですね、はい。\nA: こちら面白いですよね。<>というか穴場という感じでしょうか。ええ。 こちらからまず詳しい内容をご提案致しますね。\nB: あっはい、おねがいします。\nA: はい。 こちらはあのスズヒロという会社なんですけれども\nB: はいはいはい。\nA: スズヒロがですね、かまぼこの里というものがございまして、この中に博物館ございます。\nB: うんうん。 はい。\nA: かまぼこの原材料であったり\nB: はい。\nA: 型とかがわかるような場所になっておりまして\nB: はい。\nA: あとですね、例えば平安時代のかまぼこのレプリカがあったり\nB: うんうん、へぇー。\nA: そうなんです。あとですね、まぁ職人さんが作っている、かまぼこを作っている姿をガラス越しに見ることができたり\nB: はい。\nA: あとお客様、これどうでしょう。かまぼことかちくわを手作りで体験することもできます。\nB: ええー！そうなんですか？すごいですね。\nA: ええ。ですので、おそらく旦那様とご一緒でのご旅行とのことでしたので、お二人で、どっちがかまぼこうまく作れるかとか、おそらく奥様の方がええ、上手に作られるかと思うんですけど\nB: はい。うんうん、はい、ふふふ、そうです、いやいやいや、へー。\nA: そういったところもですね、あとその、好きな具材とかもいれることができるようなんですね。\nB: うんうん、あっ、へーすごいですね。\nA: そうですね。あとまぁ、焼きたてですので、ちくわとか、その場でも食べることができるようです。\nB: はい、うん。\nA: どうですかね、まずこれがかまぼこの博物館でして、こちらあの入館は無料となっております。\nB: そうなんですか？素晴らしいですね。へぇーはい。\nA: そうなんです。素晴らしいですよね。 かまぼこを作る体験教室はですね、お一人様1500円となっておりまして\nB: あっはいはい。\nA: まぁ、そこまであのお高いお値段ではないので\nB: うん。\nA: まぁ、あの一つ候補としてまずあの頭にとめておいていただければとおも\nB: あっ、はい。あっ、はい。はい。\nA: で、もう一つですね、次はですね、例えばお客様、先ほどの星地球博物館、こちらをご案内しましょうか？\nB: はい。はい、ええええ、はい、はいおねがいします。\nA: はい。 こちらですね、テーマがこの建物の名前にございます通り、星と地球がメインとなっておりまして\nB: はい。 はい。\nA: ま、その中で展示室が4つほどございます。\nB: はい。\nA: で、まぁ地球に関するもの生命に関するもの神奈川の自然に関するもの、そして自然との共生に関するものといった具合で\nB: うんうん。\nA: ま、その上で、まぁ恐竜とかですね。あと昆虫とかの標本とかもあるようです。\nB: へぇー、はい。\nA: で、こちらは入館料はですね、大人の方は520円お一人様でございますね、なっております。\nB: はい、はい、はい。\nA: で、最後に文学の方ですね。こちら小田原文学館と言いまして。\nB: はい。\nA: はい。北原白秋、あと尾崎加寿夫とか小田原にゆかりのある文学者の方々の原稿とか初版本が展示されているようです。\nB: あっ、うんうん。 うんうんうんうん。\nA: で、こちらは、の北原白秋に関しましては、原稿のレプリカとかも展示されているようでして。 まぁ、そういったところでは、まぁ名前はB級なんだけれども、飾られているものはA級かなという風に思います。\nB: そうですね。うんうんうん。\nA: いかがでしょうか、この文学と。\nB: はい。\nA: 地球、星、まぁ星地球博物館でございますね。それとあの、かまぼこの美味しいかまぼこ博物館。\nB: はい、はい。\nA: どちらがよろしいか、あるいは全部もあるか。\nB: そうですね。\nA: それとも二つぐらい選んでみるかいかがでしょうか？\nB: はい。もうそうですね、ちょっと欲張りなんで、全部まわりたいかななんて思ったんですけど。\nA: ええええ。\nB: できればお昼近くにかまぼこ博物館の方に伺うようにして。\nA: ええ。ええ。\nB: なんだろう。できたての、まぁ<>まずどっか一つ\nA: ええ。\nB: <>とこまわって、それからのかまぼこ博物館に行って、で自分たちが作ったかまぼこ、こう参加させていただいて、<>からの、えっと、まぁ<>行って最後に星でしめるみたいな感じに\nA: ええ、ええ、ええ。ええ、ええ。ええ。あっ。\nB: がいいかななんて思ったんですけど。\nA: いいですね。では、えーオルゴールの\nB: はい、そうですね。\nA: ところに最初アンの家に行って\nB: あっはい。\nA: はい、その次にかまぼこ博物館お昼頃行って、お召し上がりになると\nB: はい、うんうん、はい。はい。はい。\nA: その次にこちらの文学館\nB: そうですね。はい。\nA: ですね、はい。小田原文学館に行かれて最後にこちら。 星地球博物館の方に行かれるということで。か。\nB: あっはい。はい。\nA: お客様。\nB: はい。\nA: 歩いてたりすると甘いものも食べてく、食べたくなるかと思うんですけれどもいかがでしょうか。\nB: ばれました。そうですね。私もちょっと多分おそらくかまぼこ食べた後。\nA: ええ、ええ。\nB: 文学を堪能した後に絶対お腹はすくと思うんです。\nA: そうですよね。\nB: そうですね。あもうナイスなご提案でありがとう。\nA: ありがとうございます。\nB: もしいいところがあれば、はい。はい。\nA: ええ。もちろんでございます。ではお調べ致しますね。\nB: はいお願いします。\nA: 少々お待ちくださいませ。\nB: はい。\nA: お客様やはりカフェのような感じのところがいいですかね。それともお食事をまぁしっかり食べられるようなところがよろしいですか？\nB: うん、うんうん。 あっ、えっとすいません。そうですね。あの、わ、和カフェというか\nA: ええ。ええ。\nB: なんか甘味処的なところがあれば。はい。<>ますかね。はい。\nA: あっなるほど。はい。 そうでしたら、えー例えばですね、あの、二宮金次郎ってお客様ご存知ですか？\nB: えー、はい、ええええ、はい。\nA: 二宮金次郎が幼少期をこの小田原で過ごしたようで、\nB: はい。\nA: あのホウトク二宮神社っていうところがあるんですけれども。 その境内の並びにですね、和風のカフェがございます。\nB: あっ、ほんとですか。\nA: はい。名前もですね、金次郎カフェといいまして。\nB: はい。\nA: ええ、まさにこの二宮金次郎関連の、こう、テーマの、カフェとなっておりますので意外と楽しんじゃないのかなという<>\nB: はい。ええ。 うんうんうん、そうですね。はい。\nA: でわたくしどもの方のですね、情報の写真ではですね。\nB: はい、はい\nA: カプチーノであったり、あとですねこちらお食事のようなものですね。昔金次郎が食べていたっていうですね。\nB: うんうん。うん。\nA: あの食事を現代風にアレンジした。\nB: へぇー。\nA: 味噌汁のようなものもあるようで。ちょっとしたおやつ感覚ではこの金次郎カフェ結構いいかなと。\nB: あっいいですねー。ちょうど午後はかまぼこが終わった後に、その文学館に行ってるのでその<>で、はい。素晴らしいご提案だとおもいます。ありがとうございます。\nA: ええ。 ありがとうございます。\nB: はい。\nA: でですね、お客様。今のは和風だったじゃないですか。\nB: えええ。はいはい。\nA: 念のため洋風っていうのも聞いておきませんか？\nB: あっ、はい、あっそうですね。じゃあおねがいします。\nA: あのですね、名前が箱根カフェと言いまして。\nB: はい。\nA: はい。こちらですね、小田急グループのホテルがプロデュースしているカフェとなっております。\nB: はい。うんうん。はい。\nA: なので、まぁあの、なんていうんですかね。 結構その個人商店のような感じではなく、まぁ、大手がやっているカフェというふうにお考えいただければと思うんですね。\nB: はい。はい、はい。\nA: ここですね、パンであったりスイーツ、コーヒーといった洋風のものがお楽しみいただけますので。\nB: うーん。ええええ、はい。\nA: さっきの金次郎カフェで言えば和風の感じ。でこちら<>カフェであれば洋風の感じというところで。\nB: はい。はい。\nA: お客様お好み、最初は和風だったと思うんですけれども。<>ですかね。\nB: ええええ。 そうですね。和風で、最初の<>の和風でいって、で、もしあのよければ、その、えっと、洋風の方なんですけども夜は何時まであいてますか？\nA: ええ。ええ。ええ。ええ。ええ。 夜でございますね。\nB: はい\nA: こちらですね、ちょっと早くてですね、夜の七時までとなっております。\nB: そうです、そっかそっか。はい、はい。\nA: お客様、夕ご飯についてもご心配でしたら\nB: そうなんです、はい。\nA: 夕ご飯の方もですね、こちらでお調べ、別途いたしますんで。\nB: あっ、はい、はいおねがいします。\nA: <>はい、あのお昼のおやつはこちらの金次郎カフェということでよろしいでしょうか。\nB: あっ、はいおねがいします。はい。\nA: ありがとうございます。 ではですね、最後夕ご飯の方決めていきましょう。\nB: はい、はい。\nA: はい、では夕ご飯、どのようなものがよろしいでしょうか？\nB: そうですね。もう今日はずっとこう和で来てるので。\nA: ええ。\nB: えっとそのつながりで、ゆう、夕飯もちょっと洋食けっこうヘビー、ね年齢柄ヘビーになってきたので、あの。\nA: ええ、ええ。ええ、ええ。\nB: 和風で行きたいんですけどありますか？\nA: ええ。かしこまりました。\nB: はい。では、旦那様苦手なお食事等はございませんか？ あっないです。二人ともないです。はい。\nA: <>あっそうですか。ではですね、まずわたくしどもの方でご提案お店、小田原おでんというものがございます。\nB: はい、はい。\nA: まぁ、さっきかまぼこ食べたんですけれども、ええ、まさにやっぱ。\nB: ふふ。\nA: こうおでんを、結局は名物になるということですよね。<>、ええ。\nB: あっなるほど。はい。\nA: ただまぁちょっと、お昼にかまぼこ自分で作って<>\nB: やっ、いやいやいや。\nA: でも、おそらくお客様がおつくりになられるかまぼこよりもおいしいってことは絶対にここはないので。\nB: いえ。\nA: お客様が作ったものが多分一番だと思うんですね。\nB: いやいやそんなことない。 はいはい、はいいいですね。はい。\nA: これもまぁいいかとは思うんですけども。\nB: はい。\nA: ええ。えーとですね、他のお店もどんどんご提案してもいいですかね。 その中で決めていただくと。\nB: そうですね。あっ、そうですね。今すごく惹かれたんですけど、\nA: ええ。\nB: 一つだけリクエストがありまして。\nA: あっはい。もちろんです。\nB: せっかくあの小田原というか海、海側、うみそ、海側、海の近く<>ので、できればお魚を。\nA: ええ。ええ。あっ。\nB: 生ではなくて、焼き魚を堪能したいんですがありますか？\nA: ええ。 かしこまりました。それでは少々お待ちくださいませ。\nB: はい。\nA: では焼き魚でございますね。\nB: はい、ええええ。\nA: はい。 そうですね、こちらですね、小田原、やはり鮮魚のお店は多いんですけれども。\nB: ええええ。\nA: やっぱり、その鮮度を売りにしていて、生のものが結構多いかなあという感じ。\nB: そうなんですね。\nA: お客様、懐石のような感じの<>あっ、よろしいですか？\nB: いいですね。ええええいいですね。はい。\nA: はい。名前がですね、例えばだるま料理店というところがあるんですけれども。\nB: はい。ふーはい。\nA: 名前が何か楽しいですよね。\nB: 楽しいですねー。はい。\nA: で、例えばですね、お刺身と天ぷらの定食が1728円といった具合で。\nB: はい。あっはい。うんうん、はい。\nA: しかもここですね、創業が1893年。\nB: すっごーいですねー。\nA: そうですね。130年も続くお店ということで。\nB: すごーいですね。はい。\nA: で、ただここちょっと閉まるのが早くて、夜8時までなんですね。\nB: あっ、うん、あっ。うんうんうん。\nA: ですので、もうちょっと遅いところも見てみましょうか。\nB: そうですね、んっと、その、ごめんなさい、あの話を戻すようなんですけど、その、えっと。星の。\nA: ええ、ええ、はい。\nB: <>あそこが終わってすぐ行けるような感じにしたいので。\nA: ええ。\nB: その兼ね合いもあるんですけどもし間に合えばそちらでもいいですし。\nA: ええ。\nB: ひとつぐらいなんか聞いておきたいなって気持ちもするんですけど。\nA: もちろんでございます。\nB: はい。\nA: ではですね、他の。\nB: はい。\nA: お店もこちらで確認いたしますと、少々お待ちくださいね。\nB: はい。あっはい、おねがいします。\nA: はい。 やはり和風がいいんですよね。例えば\nB: そうですね。うん。\nA: 地魚を使ったフレンチとかそういうのではないということで<>\nB: あーなるほど。\nA: それもなかなか<>\nB: いいですね。地魚を使ったフレンチっていい<>ね。\nA: 生ものではないんです。\nB: ええええ。いいですね。はい。はい。\nA: なので、その点ではいいかなと思ったんですけど。\nB: あっ、おねがいします。はい教えてください。はい。\nA: はい。こちらですね、名前がイルマーレといいます。\nB: うんうんうん。はい。\nA: で、まぁ鮮魚のサラダ仕立て、まぁカルパッチョみたいなものですね。\nB: あっ、なるほど。はい。\nA: そういうものがあったりして、で、まぁ夜は8時30分まで開いています。\nB: あっ、うんうん。\nA: ただちょっとまぁ気になるのが、メニューがおまかせしかないんですね。おまかせ<>\nB: あっなるほど。うんうんうん。はい。\nA: そうしますと、例えば事前に予約とかしておいて、こういうものが食べたいですって言っておけばとても良い流れになると思うんですけれども。\nB: そうですね。うんうん。\nA: でも、なんか突然訪れても、なんか自分のイメージと違ったみたいになってしまったりすると、まぁちょっと嫌かなという気も<>\nB: あはっ、ふふふ。\nA: ま、ここに行くとしたらお電話番号をお伝えしておいて。\nB: あっ、はいおねがいします。はい。\nA: そうですね。どうですかね、お客様この。\nB: はい。\nA: フレンチのお魚の料理と。 先ほど申し上げましたような、そのだるまさんの方ですね。\nB: はい。ふふふふ。\nA: どっちがよろしいですかね？\nB: もうなんかどちらも選び、選びたい感じなんですが。\nA: ええ。\nB: <>感じなんですけど、でも、あの今ご提案くださった、フレンチ。\nA: はい。ええ。ええ。\nB: 地魚のフレンチ方は、あの予約の時に、なんと<>こちらの希望を伝えられるっておっしゃってたので。\nA: ええ、ええ、ええ。\nB: ちょっと<>しようかななんて思ったんですけ、まぁ予約をとっちゃって。\nA: あっ、あっ、ええええ。 かしこまりました。ではこちらイルマーレといいます。\nB: はい。はい。 はい。あっメモしますね。はい。\nA: はい、ありがとうございます。 ではですね、本日ご案内いたしました場所をもう一度おさらいいたします。\nB: はい。おねがいします。はい。\nA: じゃあオルゴールがアンの家ですね。\nB: はい。\nA: それと鈴廣のかまぼこ博物館。\nB: はい。はい。\nA: で、小田原の文学館。\nB: はい。<>金次郎カフェでお茶を楽しみになりまして。 はい。その次に神奈川県立生命の星地球博物館。 あっ、はい、はい。"
    llm = Llama_Swallow(model_name="./dpo-recommendation-results_2")
    summaries = llm.generate_rec_sentence_with_no_dialogue(rec_info="沖縄県本部町にあるレストランです。A級の牛肉を使用した料理や新鮮な魚介類を楽しめます。接待にもお勧めです。", temperature=1.0)
    print("Generated Text:", summaries)
    print(type(summaries))

    # 観光地推薦文の検証

    # llm = Llama_Swallow()
    # rec_info =  "北大植物園の園内にある、初代園長、宮部金吾博士に関する記念館。北海道や千島、樺太を踏破し、北方に育つ植物の研究を重ねた功績が遺品とともに展示されている。明治34年(1901)に建てられ、動植物学講堂や庁舎として使用されてきた建物そのものも歴史を静かに語る。所要30～60分くらい 女子おすすめ 雨でもOK"
    # for i in range(15):
    #     summary = llm.generate_rec_sentence_with_no_dialogue(rec_info=rec_info)
    #     print("Generated Text:", summary)
    