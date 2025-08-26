import logging
import os
import sys
from datetime import datetime

class CustomLogger:
    """
    カスタムロガークラス。ファイル名、行番号、関数名を含むログ出力を提供します。
    """
    
    def __init__(self, logger_name="app", log_level=logging.INFO, 
                 log_to_console=True, log_to_file=False, log_file_path="logs"):
        """
        ロガーの初期化
        
        Args:
            logger_name (str): ロガー名
            log_level (int): ログレベル（例：logging.DEBUG, logging.INFO）
            log_to_console (bool): コンソールに出力するかどうか
            log_to_file (bool): ファイルに出力するかどうか
            log_file_path (str): ログファイルの保存先ディレクトリ
        """
        # ロガーの取得
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # 既存のハンドラをクリア
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # フォーマットの設定
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソール出力の設定
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # ファイル出力の設定
        if log_to_file:
            # ログディレクトリが存在しない場合は作成
            if not os.path.exists(log_file_path):
                os.makedirs(log_file_path)
            
            # 日付をファイル名に含める
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = os.path.join(log_file_path, f"{logger_name}_{today}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """
        ロガーインスタンスを返します
        
        Returns:
            logging.Logger: 設定されたロガーインスタンス
        """
        return self.logger


# 使用例
if __name__ == "__main__":
    # ロガーのインスタンス化
    custom_logger = CustomLogger(
        logger_name="my_app",
        log_level=logging.DEBUG,
        log_to_console=True,
        log_to_file=True,
        log_file_path="./logs"
    )
    
    # ロガーの取得
    logger = custom_logger.get_logger()
    
    # 各種ログレベルでのログ出力例
    logger.debug("これはデバッグメッセージです")
    logger.info("これは情報メッセージです")
    logger.warning("これは警告メッセージです")
    logger.error("これはエラーメッセージです")
    logger.critical("これはクリティカルなエラーメッセージです")
    
    # 例外をログに記録する例
    try:
        1 / 0
    except Exception as e:
        logger.exception(f"例外が発生しました: {e}")