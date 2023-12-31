import logging

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import audiofile
from keys import TELEGRAM_KEY
from archdaily_summarizer import response
from archdaily_summarizer import text2speech

import subprocess
import codecs
import os

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def partition_string(message):
    max_length = 4096
    lines = message.split(". ")
    partitions = []
    current_partition = ""

    for line in lines:
        if len(current_partition + line) <= max_length:
            current_partition += line + ". "
        else:
            partitions.append(current_partition)
            current_partition = line + ". "

    if current_partition:
        partitions.append(current_partition)

    return partitions


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)

async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_message.chat_id
    """Get url and summarize the content"""
    try:
        url = str(context.args[0])
        # status = subprocess.check_output(f"python archdaily_summarizer.py --url {url}")
        # status = codecs.decode(status, 'utf-8')
        status = response(url)
        # await update.effective_message.reply_text(status)
        if len(status) > 4096:
            partitions = partition_string(status)
        else:
            partitions =[status]
        print(len(partitions))
        for partition in partitions:
            await context.bot.send_message(chat_id, text=partition)
    except:
        await update.effective_message.reply_text("Oops, please try again. Usage: /summarize URL")

async def audify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_message.chat_id
    """Get url and summarize+audify the content"""
    try:
        url = str(context.args[0])
        # status = subprocess.check_output(f"python archdaily_summarizer.py --url {url} --audify 1 --chatid {chat_id}")
        # status = codecs.decode(status, 'utf-8')
        text = response(url)
        status = text2speech(text, chat_id)
        signal, sampling_rate = audiofile.read(status, always_2d=True)
        duration = int(signal.shape[1] / sampling_rate)
        await context.bot.send_audio(chat_id=chat_id,
                                     audio=status,
                                     duration=duration,
                                     performer= 'Bot',
                                     title='Summary',
                                     caption=f"Audio duration: {duration} seconds")
        os.remove(status)
    except:
        await update.effective_message.reply_text("Oops, please try again. Usage: /audify URL")

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_KEY).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("summarize", summarize, block=False))
    application.add_handler(CommandHandler("audify", audify, block=False))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()