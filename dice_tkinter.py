''' Algorithm to play with dice: tkinter library to create GUI '''
from tkinter import Tk, ttk, Frame, Label, Button, Spinbox, messagebox, Checkbutton
import random


def get_number(char):
    '''Get value from the dice face '''

    return int(str(ord(char))[-2:]) - 55


def roll():
    ''' Create the dice and the sum '''

    try:
        d_number = int(my_list.get())
    except ValueError:
        label1.config(text='')
        label2.config(text='')
        messagebox.showerror("Wrong input", "Enter value from 1-8!")
        return
    else:
        if d_number <= 0 or d_number > 8:
            label1.config(text='')
            label2.config(text='')
            messagebox.showerror("Wrong input", "Enter value from 1-8!")
            return

    face_list = [random.choice(dice) for i in range(d_number)]
    face_numbers = list(map(get_number, face_list))
    face_total = sum(face_numbers)
    faces = ''.join(face_list)
    label1.config(text=f'{faces}')
    label2.config(text=f'You rolled a total of {face_total}')
    if face_total >= 6*d_number - 1:
        messagebox.showinfo("Critical hit", "You've dealt a big damage!")
    elif face_total <= d_number + 1:
        messagebox.showerror("Critical miss", "You've missed badly!")
        my_button.configure(state="disabled")


# defining the dice faces by the ASCII characters (unicode)
dice = ['\u2680', '\u2681', '\u2682', '\u2683', '\u2684', '\u2685']

# create tkinter instance and geometry
root = Tk()
root.title("Dice rolling")
root.geometry("900x350")

# define a frame
my_frame = Frame(root, borderwidth=2)
my_frame.pack(pady=20)

# define the input text
label0 = Label(my_frame, text='Enter number of dice:', font=('Helvetica', 15))
label0.grid(row=0, column=0)

# define the box for input
# my_entry = Entry(my_frame, width=4, font=('Helvetica', 15))
# my_entry.grid(row=1, column=0)
# my_list = Spinbox(my_frame, from_=2, to=8, width=4, font=('Helvetica', 15))
# my_list.grid(row=1, column=0)
my_list = ttk.Combobox(my_frame, width=4, values=tuple(range(1,9)), font = ("Helvetica", 10))
my_list.grid(row=1, column=0)
my_list.set(2)

# checkbutton to destroy or deactivate roll button
# my_check = Checkbutton(my_frame, )

# Dice label
label1 = Label(my_frame, font=("Helvetica", 100))
label1.grid(row=2, column=0)

# Compute and show the sum
label2 = Label(my_frame, font=("Helvetica", 15))
label2.grid(row=3, column=0, pady=0)

# Button which will trigger the rolling of dice
my_button = Button(root, text="Let's Roll!",
                   font=("Helvetica", 15), command=roll)
my_button.pack(pady=0)

my_list.focus()
root.mainloop()
