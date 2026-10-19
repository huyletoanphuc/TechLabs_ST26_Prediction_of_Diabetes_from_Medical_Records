import logging
import pandas as pd
import joblib

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters,
)

from pathlib import Path

script_dir = Path(__file__).parent

TOKEN = input("Enter your Telegram bot token: ")

logging.basicConfig(
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" , level = logging.INFO
)

model_path = script_dir / "diabetes_model.joblib"
model = joblib.load(model_path)

AGE, GENDER, BMI, GLUCOSE, INSULIN = range(5)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "👋 Welcome to the Diabetes Risk Assessment Bot!\n\n"
        "I will ask you 5 medical questions to estimate diabetes risk.\n\n"
        "Let's begin: What is the patient's Age?"
    )
    return AGE

async def get_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Age"] = float(update.message.text)
        await update.message.reply_text("What is Gender? (1 for Female, 2 for Male)")
        return GENDER
    except ValueError:
        await update.message.reply_text("Please enter a valid number for Age.")
        return AGE

async def get_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        gender = float(update.message.text)
        if gender not in [1, 2]:
            raise ValueError
        context.user_data["Gender"] = gender
        await update.message.reply_text("What is the patient's BMI? (e.g., 28.5)")
        return BMI
    except ValueError:
        await update.message.reply_text("Please enter 1 for Female or 2 for Male.")
        return GENDER

async def get_bmi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["BMI"] = float(update.message.text)
        await update.message.reply_text("What is the Glucose level? (mg/dL) (e.g., 120)")
        return GLUCOSE
    except ValueError:
        await update.message.reply_text("Please enter a valid number for BMI (mg/dL).")
        return BMI

async def get_glucose(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Glucose"] = float(update.message.text)
        await update.message.reply_text("What is the Insulin level? (mu U/ml) (e.g., 85)")
        return INSULIN
    except ValueError:
        await update.message.reply_text("Please enter a valid number for Glucose (mu U/ml).")
        return GLUCOSE

async def get_insulin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Insulin"] = float(update.message.text)

        input_data = pd.DataFrame([{
            "Age": context.user_data["Age"],
            "Gender": context.user_data["Gender"],
            "BMI": context.user_data["BMI"],
            "Glucose": context.user_data["Glucose"],
            "Insulin": context.user_data["Insulin"]
        }])

        probability = model.predict_proba(input_data)[0][1]
        threshold = 0.20  
        is_high_risk = probability >= threshold

        risk_percentage = round(probability * 100, 1)
        if is_high_risk:
            status = "⚠️ HIGH RISK DETECTED"
            recommendation = "Consult a physician for formal medical evaluation."
        else:
            status = "✅ LOW RISK"
            recommendation = "Continue maintaining healthy lifestyle practices."

        response = (
            f"📊 Assessment Summary\n"
            f"• Estimated Probability: `{risk_percentage}%`\n"
            f"• Classification: {status}\n\n"
            f"💡 Recommendation: {recommendation}\n\n"
            f"Type /start to assess another patient."
        )

        await update.message.reply_text(response, parse_mode="Markdown")
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text("Please enter a valid number for Insulin.")
        return INSULIN

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Assessment canceled. Type /start when you are ready again.")
    return ConversationHandler.END

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_age)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_gender)],
            BMI: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_bmi)],
            GLUCOSE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_glucose)],
            INSULIN: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_insulin)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)

    print("Bot is running...")
    app.run_polling()