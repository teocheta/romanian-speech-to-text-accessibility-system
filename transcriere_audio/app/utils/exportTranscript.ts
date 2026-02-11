import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { Alert } from 'react-native';

export const exportTranscriptAsPDF = async (transcript: string[]) => {
  if (transcript.length === 0) {
    Alert.alert('Nicio transcriere de exportat.');
    return;
  }

  const htmlContent = `
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Transcriere sesiune</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 24px; }
          h1 { text-align: center; color: #333; }
          p { font-size: 14px; margin-bottom: 8px; }
        </style>
      </head>
      <body>
        <h1>Transcriere sesiune</h1>
        ${transcript.map(line => `<p>${line}</p>`).join('')}
      </body>
    </html>
  `;

  try {
    const { uri } = await Print.printToFileAsync({ html: htmlContent });
    await Sharing.shareAsync(uri);
  } catch (err) {
    console.error('Eroare export PDF:', err);
    Alert.alert('Export eșuat', 'A apărut o problemă la generarea PDF-ului.');
  }
};
