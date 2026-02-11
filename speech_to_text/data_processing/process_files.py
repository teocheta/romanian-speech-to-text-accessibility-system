import pandas as pd
import os

# Căile către fișierele CSV și directoarele audio
csv_fata_mosului = "../data/transcriptions/fata_mosului.csv"
audio_dir_fata_mosului = "../split_audio_fata_mosului"

# Presupunem că ai creat un fișier CSV pentru ursul și un director separat pentru audio
csv_ursul_pacalit = "../data/transcriptions/ursul_pacalit_de_vulpe.csv" # Creează acest fișier CSV
audio_dir_ursul_pacalit = "../split_audio_ursul_pacalit_de_vulpe" # Asigură-te că ai mutat aici fișierele .wav

combined_csv_output_path = "../data/transcriptions/combined_librivox.csv"


# Funcție helper pentru a adăuga o coloană cu directorul sursă
def load_and_tag_csv(csv_path, audio_base_dir):
    df = pd.read_csv(csv_path)
    # Asigură-te că prima coloană este numele fișierului, nu calea absolută
    df.iloc[:, 0] = df.iloc[:, 0].apply(os.path.basename)
    df['audio_base_dir'] = audio_base_dir # Adăugăm o coloană care indică directorul audio
    return df

# Încărcăm și marcăm DataFrame-urile
df_fata = load_and_tag_csv(csv_fata_mosului, audio_dir_fata_mosului)
df_ursul = load_and_tag_csv(csv_ursul_pacalit, audio_dir_ursul_pacalit)

# Combinăm DataFrame-urile
combined_df = pd.concat([df_fata, df_ursul], ignore_index=True)

# Salvează fișierul CSV combinat
combined_df.to_csv(combined_csv_output_path, index=False)

print(f"Fișierul CSV combinat a fost creat la: {combined_csv_output_path}")
print(f"Total înregistrări: {len(combined_df)}")