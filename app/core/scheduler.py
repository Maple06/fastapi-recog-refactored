# Run daily dataset update and encode

from ..api.api_v1.services.facerecog_service import recogService
from apscheduler.schedulers.background import BackgroundScheduler
from .logging import logger

def dailyEncode():
    try:
        logger.info("Running daily dataset update and encoding...")
        recogService.encodeFaces()
        logger.info("Daily dataset updated and encoded!")
    except:
        logger.error("Daily dataset update and encoding failed.")

scheduler = BackgroundScheduler()
scheduler.start()

# Run the daily update every day at 01:00 a.m.
scheduler.add_job(dailyEncode, 'cron', day_of_week='mon-sun', hour=1, minute=00)