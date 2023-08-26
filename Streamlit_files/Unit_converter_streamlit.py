import streamlit as st


def convert_unit(syst, uf, ut, vf, conv_factors):

    if syst != 'temperature':
        factor_from = conv_factors[syst][uf]
        factor_to = conv_factors[syst][ut]
        value_converted = vf * factor_from / factor_to

    else:
        ts = ['C', 'F', 'K', 'R']
        convert_value = conv_factors['temperature'][uf][ts.index(ut)]
        if convert_value == 1:
            value_converted = vf
        else:
            value_converted = eval(convert_value)

    # print(f"\n{value_from} {unit_from} is equal to {value_converted:.4g} {unit_to}")

    return value_converted


conversion_factors = {
    'mass': {'g': 1, 'kg': 1e3, 'ton': 1e6, 'lb': 453.6},
    'time': {'s': 1, 'min': 60, 'h': 60**2, 'day': 24*60**2},
    'length': {'mm': 1, 'cm': 10, 'm': 1e3, 'km': 1e6, 'in': 25.4, 'ft': 25.4*12, 'mi': 1.6e6},
    'area': {'mm2': 1, 'cm2': 1e2, 'm2': 1e6, 'km2': 1e12, 'in2': 25.4**2, 'ft2': (12*25.4)**2},
    'volume': {'mm3': 1, 'cm3': 1e3, 'm3': 1e9, 'in3': 25.4**3, 'ft3': (12*25.4)**3, 'L': 1e6},
    'energy': {'J': 1, 'kJ': 1e3, 'MJ': 1e6, 'kWh': 3.6e6, 'BTU': 1055, 'cal': 4186},
    'force': {'N': 1, 'kgf': 9.81, 'lbf': 4.44},
    'power': {'W': 1, 'kW': 1e3, 'MW': 1e6, 'hp': 735, 'BTU/h': 0.293},
    'pressure': {'Pa': 1, 'kPa': 1e3, 'MPa': 1e6, 'atm': 98066.5, 'bar': 1e5,
                 'mmHg': 133.322, 'mmH2O': 9.81, 'inHg': 3386, 'psi': 6894.757},
    'temperature': {'C': [1, '9/5*value_from + 32', 'value_from + 273.15', '9/5*value_from + 32 + 459.7'],
                    'F': ['5/9*(value_from - 32)', 1, '5/9*(value_from - 32) + 273.15', 'value_from + 459.7'],
                    'K': ['value_from-273.15', '9/5*(value_from-273.15)+32', 1, '9/5*(value_from-273.15)+32+459.7'],
                    'R': ['5/9*(value_from-459.7-32)', 'unit_from-459.7', '5/9*(value_from - 459.7 - 32) + 273.15', 1]}
                     }


st.set_page_config(page_title='Unit converter', page_icon=':straight_ruler:', layout='wide')

st.markdown("""<style>.big-font {font-size:25px !important;}</style>""", unsafe_allow_html=True)

st.title(':left_right_arrow: Unit conversion tool')

system = st.selectbox('Enter unit system for conversion: ', list(conversion_factors.keys()))


col1, col2, col3, col4, _ = st.columns([1, 1, 1, 1, 2])
with col1:
    value_from = st.number_input('Enter number to convert: ')

with col2:
    unit_from = st.selectbox('Unit from: ', conversion_factors[system].keys())

with col4:
    st.write('\n')
    st.write('\n')
    button = st.button('Convert')

with col3:
    unit_to = st.selectbox('Unit to: ', conversion_factors[system].keys())

if button:
    result = convert_unit(system, unit_from, unit_to, value_from, conversion_factors)

    st.subheader('Result:')
    # st.write(f"\n{value_from} {unit_from}  is equivalent to  {result:.4g} {unit_to}")

    st.markdown(f'<p class="big-font"> {value_from} {unit_from}  is equivalent to  {result:.4g} {unit_to} </p>',
                unsafe_allow_html=True)
