import math

separator_ = '+'
separator_position_ = 10
padding_character_ = '0'
code_alphabet_ = '123456789ABCDEFGHIJK'
encoding_base_ = len(code_alphabet_)

latitude_max_ = 90
longitude_max_ = 180
max_digit_count_ = 15

pair_code_length_ = 10
pair_first_place_value_ = encoding_base_ **(pair_code_length_/2 - 1)
pair_precision_ = encoding_base_**3
pair_resolutions_ = [20.0, 1.0, .05, .0025, .000125]

grid_code_length_ = max_digit_count_ - pair_code_length_
grid_columns_ = 4
grid_rows_ = 5
grid_lat_first_place_values_ = grid_rows_**(grid_code_length_ - 1)
grid_lng_first_place_value_ = grid_columns_**(grid_code_length_ - 1)

final_lat_precision_ = pair_precision_ * grid_rows_**(max_digit_count_ - pair_code_length_)
final_lng_precision_ = pair_precision_ * grid_columns_**(max_digit_count_ - pair_code_length_)

min_trimmable_code_len_ = 6
grid_size_degrees_ = 0.000125


def clipLatitude(latitude):
    return min(90, max(-90, latitude))


def normalizaLongitude(longitude):
    while longitude < -180:
        longitude = longitude + 360
    while longitude >= 180:
        longitude = longitude - 360
    return longitude


def computeLatitudePrecision(codeLength):
    if codeLength <= 10:
        return pow(20, math.floor((codeLength / -2) + 2))
    return pow(20, -3) / pow(grid_rows_, codeLength - 10)


def encode(latitude, longitude, codeLength=pair_code_length_):
    if codeLength < 2 or (codeLength < pair_code_length_ and codeLength % 2 == 1):
        raise ValueError('Invalid Open Location Code length - ' + str(codeLength))
    codeLength = min(codeLength, max_digit_count_)
    latitude = clipLatitude(latitude)
    longitude = normalizaLongitude(longitude)
    if latitude == 90:
        latitude = latitude - computeLatitudePrecision(codeLength)
    code = ''

    latVal = int(round((latitude + latitude_max_) * final_lat_precision_, 6))
    lngVal = int(round((longitude + longitude_max_) * final_lng_precision_, 6))
    if codeLength > pair_code_length_:
        for i in range(0, max_digit_count_ - pair_code_length_):
            latDigit = latVal % grid_rows_
            lngDigit = lngVal % grid_columns_
            ndx = latDigit * grid_columns_ + lngDigit
            code = code_alphabet_[ndx] + code
            latVal //= grid_rows_
            lngVal //= grid_columns_
    else:
        latVal //= pow(grid_rows_, grid_code_length_)
        lngVal //= pow(grid_columns_, grid_code_length_)

    for i in range(0, pair_code_length_//2):
        code = code_alphabet_[lngVal % encoding_base_] + code
        code = code_alphabet_[latVal % encoding_base_] + code
        latVal //= encoding_base_
        lngVal //= encoding_base_

    return code[:separator_position_]