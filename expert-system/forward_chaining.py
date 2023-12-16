def bool_converter(option):
    if option.lower() == 'ya':
        return True
    elif option.lower() == 'tidak':
        return False
    else:
        raise Exception('Error.. Tolong pilih YA atau TIDAK!')

# Input
bertaring = bool_converter(input("Apakah hewan tersebut bertaring? [YA/TIDAK] "))
warna_bulu_berpola = bool_converter(input("Apakah hewan tersebut warna bulunya berpola? [YA/TIDAK] "))
kepala_berbulu_lebat = bool_converter(input("Apakah hewan tersebut kepalanya berbulu lebat? [YA/TIDAK] "))
berbulu_lebat = bool_converter(input("Apakah hewan tersebut berbulu lebat? [YA/TIDAK] "))
berbulu_tipis = bool_converter(input("Apakah hewan tersebut berbulu tipis? [YA/TIDAK] "))
makan_daging = bool_converter(input("Apakah hewan tersebut memakan daging? [YA/TIDAK] "))
makan_buah = bool_converter(input("Apakah hewan tersebut memakan buah? [YA/TIDAK] "))
berleher_panjang = bool_converter(input("Apakah hewan tersebut berlehar panjang? [YA/TIDAK] "))

if bertaring and makan_daging or kepala_berbulu_lebat:
    print("\nItu adalah SINGA JANTAN")
elif bertaring and makan_daging and warna_bulu_berpola:
    print("\nItu adalah MACAN TUTUL")
elif makan_buah and berbulu_lebat:
    print("\nItu adalah ORANGUTAN")
elif makan_buah and berleher_panjang and berbulu_tipis and warna_bulu_berpola:
    print("\nItu adalah JERAPAH")
elif berbulu_lebat and bertaring and warna_bulu_berpola and makan_buah:
    print("\nItu adalah PANDA")
elif berbulu_lebat and bertaring and warna_bulu_berpola and makan_buah:
    print("\nItu adalah BERUANG")
else:
    print("\nTidak ditemukan!")