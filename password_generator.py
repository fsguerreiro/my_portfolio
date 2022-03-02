from random import choices
import PySimpleGUI as sg
import string as st


def str_available():
    opt = list(st.ascii_letters + st.digits + st.punctuation)
    list(map(lambda x: opt.remove(x), ['^', '~', '`', '.', ',', "'", '"']))
    return opt


def check_number(password):
    if password in st.digits:
        return True
    return False


def check_lower(password):
    if password in st.ascii_lowercase:
        return True
    return False


def check_upper(password):
    if password in st.ascii_uppercase:
        return True
    return False


def check_character(password):
    if password in st.punctuation:
        return True
    return False


def pass_generate(num):
    while True:
        passw1 = choices(str_available(), k=num)
        if any(list(filter(check_number, passw1))) is False:
            continue
        if any(list(filter(check_upper, passw1))) is False:
            continue
        if any(list(filter(check_lower, passw1))) is False:
            continue
        if any(list(filter(check_character, passw1))) is False:
            continue
        passw = ''.join(passw1)
        return passw


def save_password(password_real, values):
    with open('passwords.txt', 'a') as file:
        file.write(f"Site/App: {values['site']} \nUser/e-mail: {values['user']} \nPassword: {password_real}\n\n")


layout = [
    [sg.Text('Site/App: ', key='site_text', size=(11, 1)), sg.InputText(key='site', size=(30, 1))],
    [sg.Text('User/e-mail: ', key='user_text', size=(11, 1)), sg.InputText(key='user', size=(30, 1))],
    [sg.Text('Enter number of characters for the password:'), sg.Combo(list(range(12, 25)), default_value=12, key='Size')],
    [sg.Output(size=(42, 4), key='Copy')],
    [sg.Button('Create Password'), sg.Button('Clear Output')]
            ]

window = sg.Window('Password Generator', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == 'Create Password':
        password_real = pass_generate(num=values['Size'])
        print('Site/App:', values['site'])
        print('User/e-mail:', values['user'])
        print('Password:', password_real)
        save_password(password_real, values)
    if event == 'Clear Output':
        print('\n'*100)

window.close()
