from random import choices
import string
import streamlit as st


class Password:

    def __init__(self, want_digits=True, want_special=True):

        self.want_digits = want_digits
        self.want_special = want_special
        self.special_chr = '!@#$%&*_-+=<>:;?'
        self.chars = string.ascii_letters
        self.chars += string.digits if self.want_digits else ''
        self.chars += self.special_chr if self.want_special else ''

    def create_pwd(self, length):
        meet_requirement = False

        while not meet_requirement:

            pwd = ''.join(choices(self.chars, k=length))
            has_lower = any(c.islower() for c in pwd)
            has_upper = any(c.isupper() for c in pwd)
            check_req = [has_lower, has_upper]

            if self.want_digits:
                has_number = any(c.isdigit() for c in pwd)
                check_req.append(has_number)

            if self.want_special:
                has_special = any(c in self.special_chr for c in pwd)
                check_req.append(has_special)

            meet_requirement = all(check_req)

        return pwd


st.set_page_config(page_title='Password generator', page_icon=':memo:', layout='wide')

st.title('Password generator tool')

col1, col2, col3 = st.columns([1, 1, 1], gap='large')
with col1:
    len_pass = st.slider('Select password length: ', 8, 24, 8, 1)

with col2:
    w_digits = st.checkbox('Include digits')

with col3:
    w_special = st.checkbox('Include special characters')

if st.button('Create password'):
    final_pwd = Password(w_digits, w_special).create_pwd(len_pass)
    st.write(f':heavy_check_mark: **Your password is {final_pwd}**')

