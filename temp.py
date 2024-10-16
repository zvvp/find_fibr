from functions import parse_B_txt


r_pos, intervals, chars, forms = parse_B_txt()
print(r_pos.size)
print(intervals.size)
print(chars.size)
print(forms.size)