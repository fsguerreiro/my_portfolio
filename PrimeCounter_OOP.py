from math import isqrt

class Prime:

    def __init__(self, n):
        self.num = n
        self.prime_list = []


    def prime_checklist(self):
        for p in range(2, self.num + 1):

            if self.is_prime(p):
                self.prime_list.append(p)

        self.soma_primos = sum(self.prime_list)
        self.qtdade = len(self.prime_list)


    def is_prime(self, num):
        if num == 2:
            return True

        if num < 2 or num % 2 == 0:
            return False

        for divisor in range(3, isqrt(num) + 1, 2):

            if num % divisor == 0:
                return False

        return True


    def is_prime_num(self):

        if self.is_prime(self.num):
            print(f"The number {self.num} is prime.")

        else:
            print(f"The number {self.num} is not prime.")

n = 1000
lista = Prime(n)
lista.prime_checklist()
lista.is_prime_num()
print(f"\nThe prime numbers until {lista.num} are: ", end='')
print(*lista.prime_list, sep=', ')
print(f"\nThe sum of these primes is {lista.soma_primos}")
