const API_BASE_URL = 'http://192.168.0.30:5000'; // IP local sau domeniu

export const uploadAudioFile = async (
  uri: string,
  sessionId?: string,
  model: 'custom' | 'whisper' = 'custom'
): Promise<string | null> => {
  const fileName = uri.split('/').pop() || 'audio.m4a';

  const formData = new FormData();
  formData.append('file', {
    uri,
    name: fileName,
    type: getMimeType(fileName),
  } as any);

  if (sessionId) {
    formData.append('session', sessionId);
    console.log('Trimitem session ID:', sessionId);
  }

  formData.append('model', model);
  console.log('Model ales:', model);

  console.log('Trimit fișierul la server:', uri);

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Eroare backend:', errorText);
      return null;
    }

    const result = await response.json();
    console.log('Răspuns server:', result);
    return result.transcript || null;
  } catch (err) {
    console.error('Eroare la trimiterea fișierului:', err);
    return null;
  }
};

const getMimeType = (fileName: string): string => {
  const ext = fileName.split('.').pop()?.toLowerCase();

  switch (ext) {
    case 'wav':
      return 'audio/wav';
    case 'mp3':
      return 'audio/mpeg';
    case 'm4a':
      return 'audio/m4a';
    default:
      return 'application/octet-stream'; // fallback sigur
  }
};

export const getLiveTranscript = async (sessionId: string): Promise<string> => {
  try {
    const response = await fetch(`${API_BASE_URL}/transcript/${sessionId}`);
    const data = await response.json();
    return data.transcript || '';
  } catch (err) {
    console.error('Eroare la fetch transcript:', err);
    return '';
  }
};

export const generateSessionId = (length = 4): string => {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
};

