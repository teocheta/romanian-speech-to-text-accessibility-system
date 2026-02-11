from collections import defaultdict


def build_vocab(transcripts):
    """
    Creează un vocabular de caractere din lista de transcrieri.
    Returnează două dicționare: char2idx și idx2char.
    """
    characters = set("".join(transcripts).lower())
    vocab = sorted(list(characters))
    vocab.append('_')  # simbolul blank pentru CTC
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return char2idx, idx2char


def encode_transcript(transcript, char2idx):
    # Asigură-te că toate transcrierile sunt în litere mici
    transcript_lower = transcript.lower()

    # Construiește un set de caractere permise din vocabularul furnizat
    allowed_chars_set = set(char2idx.keys())
    allowed_chars_set.discard('_') # Caracterul blank nu este în textul propriu-zis

    encoded = []
    for char in transcript_lower:
        if char in allowed_chars_set:
            encoded.append(char2idx[char])
        # else:
            # print(f"Avertisment: Caracterul '{char}' nu este în vocabular și va fi ignorat.")
    return encoded


def decode_prediction(indices, idx2char, blank_idx):
    """
    Decodează secvența de indici returnată de modelul CTC.
    Elimină caractere duplicate consecutive și simboluri blank.
    """
    decoded = []
    previous_idx = None

    for idx in indices:
        if idx == blank_idx:
            previous_idx = None  # resetăm pentru a putea relua același caracter după blank
            continue

        if idx != previous_idx:
            char = idx2char.get(idx, '')
            decoded.append(char)
        previous_idx = idx

    text = ''.join(decoded)

    # Curățare opțională
    text = text.replace("  ", " ").strip()
    return text






