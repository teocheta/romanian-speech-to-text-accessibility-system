import { Image } from 'expo-image';
import { Link } from 'expo-router';
import { StyleSheet } from 'react-native';

import ParallaxScrollView from '@/components/ParallaxScrollView';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

export default function HomeScreen() {
  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#A1CEDC', dark: '#1D3D47' }}
      headerImage={
        <Image
          source={require('@/assets/images/partial-react-logo.png')}
          style={styles.reactLogo}
        />
      }>
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Bine ai venit!</ThemedText>
        <ThemedText>Cu ce te putem ajuta?</ThemedText>
      </ThemedView>

      <ThemedView style={styles.roleContainer}>
        <ThemedText type="subtitle">Profesor</ThemedText>
        <Link href="/teacher/live-rec-trans-chunks">
          <ThemedText type="link">Înregistrare lecție</ThemedText>
        </Link>
      </ThemedView>

      <ThemedView style={styles.roleContainer}>
        <ThemedText type="subtitle">Elev</ThemedText>
        <Link href="/student/live-transcription">
          <ThemedText type="link">Vezi transcriere în timp real</ThemedText>
        </Link>
        <Link href="/student/load-file">
          <ThemedText type="link">Încarcă fișier audio</ThemedText>
        </Link>
      </ThemedView>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 40,
    gap: 8,
  },
  roleContainer: {
    marginBottom: 32,
    paddingHorizontal: 20,
    gap: 8,
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: 'absolute',
  },
});
