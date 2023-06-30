from random import choice

class HangmanGame:

    def __init__(self):
        self.word_list = ["valdeci", "adamastor", "jurandir", "noemia", "geraldo", "otacilio",
                          "agenor", "valmor", "francisca", "antonia", "raimunda", "matilde", "januario"]
        self.word = choice(self.word_list)
        self.lt = list(self.word)
        self.display = ["__" for _ in range(len(self.lt))]
        self.letters = []
        self.count = 0


    def play(self, tries):
        print(' '.join(self.display), f"({len(self.lt)} letters)")

        while self.count < tries:
            print(f"\nYou have {tries - self.count} guesses remaining.", end=' ')
            print(f"Letters tried: {self.letters}")
            self.count += 1
            guess = input("\nGuess a letter: ").lower()
            self.letters.append(guess)

            correct_guess = False
            for i in range(len(self.lt)):
                if self.lt[i] == guess:
                    self.display[i] = guess
                    correct_guess = True

            print(' '.join(self.display))

            if not ('__' in self.display):
                print("\nCongrats! You've won!")
                return

            if not correct_guess:
                print("\nIncorrect guess!")

            if self.count == tries:
                print(f"\nLetters tried: {self.letters}")
                final_guess = input("\nWhat's your final guess? ")
                if final_guess.lower() == self.word:
                    print("\nCongrats! You've won!")
                    return
                else:
                    print(f"\nSorry, you've failed! The word was {self.word}")
                    return

        print(f"\nYour guesses are over! The word was {self.word}")


while True:
    game = HangmanGame()
    game.play(tries = 6)
    play_again = input("\nPlay again (y/n)? ")

    if play_again == 'n':
        break

