#requires pronouncing library
import pronouncing

# drawn from Welzey Sherman's "Ghost Writing with TensorFlow" blog
# which references "Evaluating Creative Language Generation: The Case of Rap Lyric Ghostwriting" by Potash, Romanov, and Rumshisky
# url: https://towardsdatascience.com/ghost-writing-with-tensorflow-49e77e26978f

''' Rhyme density is calculated by taking the number of rhymed syllables and divide it by total number of syllables'''

def calc_rhyme_density(bars):
  total_syllables = 0
  rhymed_syllables = 0
  for bar in bars:
    for word in bar.split():
      p = pronouncing.phones_for_word(word)
      if len(p) == 0:
        break
      syllables = pronouncing.syllable_count(p[0])
      total_syllables += syllables
      has_rhyme = False
      for rhyme in pronouncing.rhymes(word):
        if has_rhyme:
          break
        for idx, r_bar in enumerate(bars):
          if idx > 4:
            break
          if rhyme in r_bar:
            rhymed_syllables += syllables
            has_rhyme = True
            break
  return rhymed_syllables/total_syllables