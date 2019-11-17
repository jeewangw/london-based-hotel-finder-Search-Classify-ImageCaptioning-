import re

the_string = "New Cars, Used Cars, Car Reviews, Car Finance Advice - Cars.com"
the_string = re.sub(r'(car)', r'<b>\1</b>', the_string, flags=re.I)
print(the_string)

