import React, { useEffect, useState } from 'react';
import { Button, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { getLiveTranscript } from '../../api/backend';

export default function LiveTranscriptionScreen() {
const [sessionId, setSessionId] = useState('');
const [transcript, setTranscript] = useState('');
const [active, setActive] = useState(false);
const [error, setError] = useState<string | null>(null);


useEffect(() => {
  if (!active || !sessionId) return;

  const interval = setInterval(async () => {
    try {
      const data = await getLiveTranscript(sessionId);

      if (!data || data.trim() === '') {
        setError('Sesiunea nu a fost găsită sau nu conține transcriere.');
        setTranscript('');
      } else {
        setError(null);
        setTranscript(data);
      }
    } catch (err) {
      console.error('Eroare la fetch transcript:', err);
      setError('Eroare la conectarea cu serverul.');
    }
  }, 4000);

  return () => clearInterval(interval);
}, [active, sessionId]);


  return (
    <SafeAreaView style={styles.safeContainer}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
      <Text style={styles.title}>Transcriere live (elev)</Text>
      <TextInput
              placeholder="Introdu ID sesiune"
              value={sessionId}
              onChangeText={setSessionId}
              style={styles.input}
      />

      <Button
          title="Conectează-te"
          onPress={() => setActive(true)}
          disabled={!sessionId}
        />
      <Button
        title="Deconectează-te"
        color="crimson"
        onPress={() => {
          setActive(false);
          setTranscript('');
          setSessionId('');
          setError(null);
        }}
      />

        {active && (
        <Text style={styles.connectedText}>
          Conectat la sesiunea: {sessionId}
        </Text>
          )}

        {error && (
          <Text style={styles.errorText}>
                {error}
          </Text>
          )}
        {active && (
          <View style={styles.transcriptBox}>
          {transcript
            ? transcript.split('\n').map((line, idx) => (
                <Text key={idx} style={styles.line}>{line}</Text>
              ))
            : <Text style={styles.line}>Se încarcă transcrierea...</Text>}
        </View>
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
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  transcriptBox: {
    backgroundColor: '#f1f1f1',
    borderRadius: 10,
    padding: 15,
  },
  line: {
    fontSize: 16,
    marginBottom: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    padding: 10,
    marginBottom: 10,
    fontSize: 16,
  },
  connectedText: {
    fontSize: 14,
    color: 'green',
    textAlign: 'center',
    marginVertical: 10,
  },
  errorText: {
    color: 'red',
    textAlign: 'center',
    marginTop: 10,
  },
  
  
});
