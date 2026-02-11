import { Audio } from 'expo-av';
import React, { useRef, useState } from 'react';
import {
  Alert,
  Button,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { generateSessionId, uploadAudioFile } from '../../api/backend';
import { exportTranscriptAsPDF } from '../../utils/exportTranscript';

export default function LiveTranscriptionProfesorScreen() {
  const [transcript, setTranscript] = useState<string[]>([]);
  const [isRecordingState, setIsRecordingState] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isRecordingRef = useRef<boolean>(false);
  const sessionRef = useRef<string | null>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const timeRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [selectedModel, setSelectedModel] = useState<'custom' | 'whisper'>('custom');

  const startRecordingLoop = async () => {
    try {
      setIsRecordingState(true);
      setRecordingTime(0);
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      await recording.startAsync();
      recordingRef.current = recording;
      isRecordingRef.current = true;

      timeRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      intervalRef.current = setInterval(() => {
        recordNextChunk();
      }, 4000);
    } catch (err) {
      console.error('Eroare la inițializarea înregistrării:', err);
      setIsRecordingState(false);
      Alert.alert('Eroare', 'Nu am putut porni înregistrarea.');
    }
  };

  const recordNextChunk = async () => {
  const oldRecording = recordingRef.current;
  if (!oldRecording || !isRecordingRef.current) return;

  try {
    try {
      await oldRecording.stopAndUnloadAsync();
    } catch (err) {
      console.warn('Înregistrarea era deja descărcată (chunk):', err);
    }

    const uri = oldRecording.getURI();
    recordingRef.current = null;

    if (uri) {
      const text = await uploadAudioFile(uri, sessionRef.current ?? '', selectedModel);
      if (text) {
        setTranscript(prev => [...prev, text]);
      }
    }

    const { recording: newRecording } = await Audio.Recording.createAsync(
      Audio.RecordingOptionsPresets.HIGH_QUALITY
    );
    await newRecording.startAsync();
    recordingRef.current = newRecording;
  } catch (err) {
    console.error('Eroare în recordNextChunk:', err);
  }
};


  const stopRecordingLoop = async () => {
  try {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timeRef.current) clearInterval(timeRef.current);
    isRecordingRef.current = false;
    setIsRecordingState(false);

    const recording = recordingRef.current;
    if (recording) {
      try {
        await recording.stopAndUnloadAsync();
      } catch (err) {
        console.warn('Înregistrarea era deja descărcată (stop):', err);
      }

      const uri = recording.getURI();
      recordingRef.current = null;

      if (uri) {
        const lastChunk = await uploadAudioFile(uri, sessionRef.current ?? '', selectedModel);
        if (lastChunk) {
          setTranscript(prev => [...prev, lastChunk]);
        }
      }
    }
  } catch (err) {
    console.error('Eroare la oprirea înregistrării:', err);
  }
};


  const handleStartSession = () => {
  // Oprește orice interval anterior
  if (intervalRef.current) clearInterval(intervalRef.current);
  if (timeRef.current) clearInterval(timeRef.current);

  // Închide orice înregistrare activă rămasă
  if (recordingRef.current) {
    try {
      recordingRef.current.stopAndUnloadAsync();
    } catch (err) {
      console.warn('Nu am putut închide înregistrarea anterioară (probabil era deja închisă):', err);
    }
    recordingRef.current = null;
  }

  isRecordingRef.current = false;
  setIsRecordingState(false);
  setTranscript([]);
  setRecordingTime(0);

  // Generează o nouă sesiune
  const newId = generateSessionId();
  setSessionId(newId);
  sessionRef.current = newId;

  console.log('Sesiune nouă inițiată:', newId);
};

  const handleEndSession = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timeRef.current) clearInterval(timeRef.current);
    isRecordingRef.current = false;
    setIsRecordingState(false);
    setTranscript([]);
    setSessionId(null);
    sessionRef.current = null;
    setRecordingTime(0);
    Alert.alert('Sesiune încheiată');
  };

  return (
    <SafeAreaView style={styles.safeContainer}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <Text style={styles.title}>Înregistrare și transcriere live</Text>

        <Button
          title="Start sesiune"
          onPress={handleStartSession}
          disabled={sessionId !== null}
        />

        {sessionId && (
          <>
            <Text style={styles.sessionId}>Cod sesiune: {sessionId}</Text>

            <View style={{ marginVertical: 10 }}>
              <Text style={{ fontSize: 16, fontWeight: 'bold' }}>Alege modelul de transcriere:</Text>
              <View style={{ flexDirection: 'row', marginTop: 8 }}>
                <Button
                  title="Model propriu"
                  onPress={() => setSelectedModel('custom')}
                  color={selectedModel === 'custom' ? '#007AFF' : '#ccc'}
                />
                <View style={{ width: 10 }} />
                <Button
                  title="Whisper"
                  onPress={() => setSelectedModel('whisper')}
                  color={selectedModel === 'whisper' ? '#007AFF' : '#ccc'}
                />
              </View>
              <Text style={{ textAlign: 'center', fontStyle: 'italic', marginTop: 6 }}>
                Model activ: {selectedModel === 'custom' ? 'Model propriu' : 'Whisper'}
              </Text>
            </View>
          </>
        )}

        {isRecordingState && (
          <View style={styles.recordingIndicator}>
            <View style={styles.dot} />
            <Text style={styles.recordingText}>
              Se înregistrează... {Math.floor(recordingTime / 60)}:{('0' + (recordingTime % 60)).slice(-2)}
            </Text>
          </View>
        )}

        <Button
          title="Start înregistrare"
          onPress={startRecordingLoop}
          disabled={!sessionId || isRecordingState}
        />
        <Button
          title="Stop înregistrare"
          onPress={stopRecordingLoop}
          disabled={!isRecordingState}
        />
        <Button
          title="Încheie sesiunea"
          onPress={handleEndSession}
          disabled={!sessionId || isRecordingState}
          color="crimson"
        />

        <View style={styles.transcriptBox}>
          {transcript.map((line, index) => (
            <Text key={index} style={styles.line}>{line}</Text>
          ))}
        </View>

        <Button
          title="Exportă ca PDF"
          onPress={() => exportTranscriptAsPDF(transcript)}
          disabled={transcript.length === 0}
        />
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
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  transcriptBox: {
    marginTop: 30,
    backgroundColor: '#f1f1f1',
    borderRadius: 10,
    padding: 15,
  },
  line: {
    fontSize: 16,
    marginBottom: 10,
  },
  sessionId: {
    fontSize: 14,
    color: '#888',
    marginBottom: 10,
    textAlign: 'center',
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: 'red',
    marginRight: 8,
    shadowColor: 'red',
    shadowRadius: 6,
    shadowOpacity: 0.8,
    shadowOffset: { width: 0, height: 0 },
  },
  recordingText: {
    fontSize: 14,
    color: 'red',
    fontWeight: 'bold',
  },
});
