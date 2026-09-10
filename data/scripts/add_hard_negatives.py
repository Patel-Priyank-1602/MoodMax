"""
Add curated hard negative examples to the training dataset.

Hard negatives are realistic, non-emotional, factual declarative sentences
(schedules, meetings, weather, technical facts, instructions) explicitly labeled
as neutral=1 and 0 for all other emotions.

This prevents the model from treating factual statements (like "The meeting starts at ten")
as fear or sadness simply because they lack overtly positive emotion keywords.
"""

import argparse
from pathlib import Path
import pandas as pd

TARGET_CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

HARD_NEGATIVES = [
    # Scheduling & Meetings
    {"text": "The meeting starts at ten.", "lang": "en"},
    {"text": "The project status call is scheduled for Thursday at 2:00 PM.", "lang": "en"},
    {"text": "Our weekly team sync will take place in Conference Room B.", "lang": "en"},
    {"text": "The deadline for the report submission is tomorrow afternoon.", "lang": "en"},
    {"text": "The webinar begins promptly at eleven o'clock.", "lang": "en"},
    {"text": "The office will be closed on Monday for the federal holiday.", "lang": "en"},
    {"text": "The morning standup is moved to 9:30 AM.", "lang": "en"},
    {"text": "The calendar invite for the quarterly review has been sent.", "lang": "en"},
    {"text": "The presentation will last approximately forty-five minutes.", "lang": "en"},
    {"text": "Appointments are available between nine in the morning and five in the evening.", "lang": "en"},

    # Workplace & Administrative Statements
    {"text": "Please find the attached spreadsheet with last quarter's figures.", "lang": "en"},
    {"text": "The document has been saved to the shared team directory.", "lang": "en"},
    {"text": "All employee badge renewals will be processed at reception.", "lang": "en"},
    {"text": "The cafeteria serves lunch from noon until two.", "lang": "en"},
    {"text": "A confirmation email was dispatched to your registered address.", "lang": "en"},
    {"text": "The software update will be installed automatically tonight.", "lang": "en"},
    {"text": "Please submit your time sheets before five on Friday.", "lang": "en"},
    {"text": "The printer on the third floor is currently refilling paper.", "lang": "en"},
    {"text": "Here is the summary of items discussed in today's agenda.", "lang": "en"},
    {"text": "The contract was signed by both parties earlier this morning.", "lang": "en"},

    # Weather & Physical Environment
    {"text": "The current temperature outside is twenty-two degrees Celsius.", "lang": "en"},
    {"text": "Tomorrow's forecast predicts partly cloudy skies with light westerly winds.", "lang": "en"},
    {"text": "Barometric pressure remains steady across the region.", "lang": "en"},
    {"text": "Sunrise was at 6:12 AM and sunset will be at 7:45 PM.", "lang": "en"},
    {"text": "Average rainfall for this month is measured at sixty millimeters.", "lang": "en"},
    {"text": "The humidity level is forty-eight percent today.", "lang": "en"},

    # Transit & Schedules
    {"text": "The next commuter train departs from platform three in ten minutes.", "lang": "en"},
    {"text": "Flight 342 has arrived at gate twelve.", "lang": "en"},
    {"text": "Buses on this route operate at fifteen-minute intervals.", "lang": "en"},
    {"text": "The subway line operates between five in the morning and midnight.", "lang": "en"},
    {"text": "Road construction is scheduled on Highway 101 between mile markers 14 and 18.", "lang": "en"},

    # Science & Factual Declaratives
    {"text": "Water boils at one hundred degrees Celsius at sea level.", "lang": "en"},
    {"text": "The Earth completes one rotation on its axis approximately every twenty-four hours.", "lang": "en"},
    {"text": "Nitrogen accounts for seventy-eight percent of the Earth's atmosphere.", "lang": "en"},
    {"text": "Light travels at approximately three hundred thousand kilometers per second.", "lang": "en"},
    {"text": "The human skeleton consists of two hundred and six bones.", "lang": "en"},

    # Technical & Routine Notifications
    {"text": "Your account balance is three hundred and fifty dollars.", "lang": "en"},
    {"text": "The system restart is required to finish applying the patch.", "lang": "en"},
    {"text": "Battery charge level is currently sixty-four percent.", "lang": "en"},
    {"text": "The file size is twenty-eight megabytes.", "lang": "en"},
    {"text": "Press one for English or press two for customer service.", "lang": "en"},

    # Multilingual Factual Declaratives (Hindi, Spanish, French, German)
    {"text": "बैठक सुबह दस बजे शुरू होगी।", "lang": "hi"},
    {"text": "कल सुबह बारिश होने की संभावना है।", "lang": "hi"},
    {"text": "ट्रेन प्लेटफार्म नंबर चार पर पहुंच चुकी है।", "lang": "hi"},
    {"text": "कार्यालय का समय सुबह नौ से शाम छह बजे तक है।", "lang": "hi"},
    {"text": "दस्तावेज़ साझा फ़ोल्डर में सहेजा गया है।", "lang": "hi"},
    {"text": "La reunión comienza a las diez de la mañana.", "lang": "es"},
    {"text": "El tren sale del andén número dos cada veinte minutos.", "lang": "es"},
    {"text": "El pronóstico para mañana es soleado y templado.", "lang": "es"},
    {"text": "El informe trimestral está disponible en el servidor.", "lang": "es"},
    {"text": "La réunion commence à dix heures.", "lang": "fr"},
    {"text": "Le train arrive sur la voie numéro trois.", "lang": "fr"},
    {"text": "Das Treffen beginnt um zehn Uhr.", "lang": "de"},
    {"text": "Der Zug fährt von Gleis vier ab.", "lang": "de"},
]


def add_hard_negatives(data_dir: Path = None, output_filename: str = "train_collapsed.csv", repeat: int = 5):
    """Inject hard negative examples into the training set."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "processed"

    train_path = data_dir / "train_collapsed.csv"
    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        return

    df_train = pd.read_csv(train_path)
    print(f"Loaded existing training set: {len(df_train):,} rows")

    # Create records for hard negatives
    new_rows = []
    # Repeat each hard negative multiple times to give them sufficient weight in gradient updates
    for _ in range(repeat):
        for idx, item in enumerate(HARD_NEGATIVES):
            row = {
                "id": f"hard_neg_{idx}_{_}",
                "text": item["text"],
                "lang": item.get("lang", "en"),
                "joy": 0,
                "sadness": 0,
                "anger": 0,
                "fear": 0,
                "surprise": 0,
                "disgust": 0,
                "neutral": 1,
                "sentiment": "Neutral",
            }
            new_rows.append(row)

    df_neg = pd.DataFrame(new_rows)
    print(f"Adding {len(df_neg):,} hard negative instances ({len(HARD_NEGATIVES)} unique x {repeat} repetitions)...")

    # Combine and shuffle
    df_combined = pd.concat([df_train, df_neg], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_path = data_dir / output_filename
    df_combined.to_csv(out_path, index=False)
    print(f"[OK] Saved updated training dataset with hard negatives to: {out_path}")
    print(f"Total training rows: {len(df_combined):,}")
    print(f"Neutral count: {df_combined['neutral'].sum():,} ({df_combined['neutral'].sum()/len(df_combined)*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add neutral hard negatives to training set")
    parser.add_argument("--repeat", type=int, default=5, help="Number of times to replicate the hard negative set")
    parser.add_argument("--output", type=str, default="train_collapsed.csv", help="Output CSV filename")
    args = parser.parse_args()

    add_hard_negatives(repeat=args.repeat, output_filename=args.output)
