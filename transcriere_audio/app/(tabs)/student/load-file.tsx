import * as DocumentPicker from 'expo-document-picker';
import React, { useRef, useState } from 'react';
import {
  Alert,
  Button,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { generateSessionId, uploadAudioFile } from '../../api/backend';

export default function LoadFileScreen() {
  const [fileInfo, setFileInfo] = useState<DocumentPicker.DocumentPickerResult | null>(null);
  const [webFile, setWebFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const pickFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'audio/*',
        copyToCacheDirectory: true,
      });

      if (!result.canceled) {
        setFileInfo(result);
        console.log('Fișier selectat:', result);
      }
    } catch (err) {
      Alert.alert('Eroare', 'A apărut o problemă la selectarea fișierului.');
      console.error(err);
    }
  };

  const handleWebFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setWebFile(file);
      console.log('Fișier web selectat:', file.name);
    }
  };

  const handleWebUpload = async () => {
    if (!webFile || !sessionId) {
      Alert.alert('Eroare', 'Selectează un fișier și generează un cod de sesiune.');
      return;
    }

    const formData = new FormData();
    formData.append('file', webFile);
    formData.append('session', sessionId);
    console.log('🔑 Session ID:', sessionId);

    try {
      const response = await fetch('http://192.168.0.30:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const raw = await response.text();
      console.log('Răspuns brut:', raw);

      const result = JSON.parse(raw);
      console.log('JSON parsat:', result);

      if (Platform.OS === 'web') {
        window.alert('Transcriere:\n' + result.transcript);
      } else {
        Alert.alert('Transcriere', result.transcript || 'Nu am primit transcriere.');
      }
    } catch (err) {
      console.error('Eroare la trimitere (web):', err);
      Alert.alert('Eroare', 'Trimiterea a eșuat.');
    }
  };

  return (
    <SafeAreaView style={styles.safeContainer}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <Button
          title="Generează cod sesiune"
          onPress={() => {
            const id = generateSessionId();
            setSessionId(id);
            console.log('🆔 Cod sesiune generat:', id);
          }}
          color="#007AFF"
        />

        {sessionId && (
          <Text style={{ textAlign: 'center', marginVertical: 10, fontSize: 16, fontWeight: 'bold', color: '#4CAF50' }}>
            Cod sesiune: {sessionId}
          </Text>
        )}

        {Platform.OS === 'web' ? (
          <>
            <Text style={styles.title}>Încarcă fișier audio (web)</Text>
            <input
              type="file"
              accept="audio/*"
              ref={fileInputRef}
              style={{ marginBottom: 20 }}
              onChange={handleWebFileChange}
            />
            {webFile && sessionId && (
              <View style={styles.fileBox}>
                <Text style={styles.label}>Fișier selectat:</Text>
                <Text>{webFile.name}</Text>
                <Button title="Trimite fișier pentru transcriere" onPress={handleWebUpload} />
              </View>
            )}
          </>
        ) : (
          <>
            <Text style={styles.title}>Încarcă fișier audio</Text>
            <Button title="Selectează fișier audio" onPress={pickFile} />

            {fileInfo && !fileInfo.canceled && sessionId && (
              <View style={styles.fileBox}>
                <Text style={styles.label}>Nume fișier:</Text>
                <Text>{fileInfo.assets?.[0]?.name}</Text>

                <Button
                  title="Trimite fișier pentru transcriere"
                  onPress={async () => {
                    const uri = fileInfo.assets[0].uri;
                    const transcript = await uploadAudioFile(uri, sessionId);
                    Alert.alert('Transcriere', transcript || 'Nu am primit transcriere.');
                  }}
                />
              </View>
            )}
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollContainer: {
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    fontSize: 22,
    marginBottom: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  fileBox: {
    marginTop: 30,
    backgroundColor: '#f1f1f1',
    padding: 15,
    borderRadius: 10,
  },
  label: {
    fontWeight: 'bold',
    marginTop: 10,
  },
  path: {
    fontSize: 12,
    color: '#555',
  },
});
