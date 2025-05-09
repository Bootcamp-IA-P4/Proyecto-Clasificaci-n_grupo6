import db_manager as db
import db_models as db_mod
import datetime

connection = db.db_connection()

prediction = db_mod.Prediction(
    timestamp=datetime.now(),
    prediction=True,
    prediction_rate=0.85,
    page_values=5,
    exit_rates=0.12,
    bounce_rates=0.34,
    weekend=False,
    administrative=2,
    informational=1,
    product_related=4,
    administrative_duration=120,
    informational_duration=60,
    product_related_duration=300,
    month=5,
    new_visitor=True )

db.save_prediction(prediction, connection)