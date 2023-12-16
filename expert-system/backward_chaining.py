hewan = ('singa jantan', 'macan tutul', 'orangutan', 'jerapah', 'panda', 'beruang')

def bool_converter(option):
    if option.lower() == 'ya':
        return True
    elif option.lower() == 'tidak':
        return False
    else:
        raise Exception('Error.. Tolong pilih YA atau TIDAK!')

# Input
nama = str(input("Pilih hewan berikut? ['Singa Jantan', 'Macan Tutul', 'Orangutan', 'Jerapah', 'Panda', 'Beruang'] "))

if nama in hewan:
    bertaring = bool_converter(input("Apakah hewan tersebut bertaring? [YA/TIDAK] "))
    warna_bulu_berpola = bool_converter(input("Apakah hewan tersebut warna bulunya berpola? [YA/TIDAK] "))
    kepala_berbulu_lebat = bool_converter(input("Apakah hewan tersebut kepalanya berbulu lebat? [YA/TIDAK] "))
    berbulu_lebat = bool_converter(input("Apakah hewan tersebut berbulu lebat? [YA/TIDAK] "))
    berbulu_tipis = bool_converter(input("Apakah hewan tersebut berbulu tipis? [YA/TIDAK] "))
    makan_daging = bool_converter(input("Apakah hewan tersebut memakan daging? [YA/TIDAK] "))
    makan_buah = bool_converter(input("Apakah hewan tersebut memakan buah? [YA/TIDAK] "))
    berleher_panjang = bool_converter(input("Apakah hewan tersebut berlehar panjang? [YA/TIDAK] "))
else:
    raise Exception('Error.. Tolong pilih YA atau TIDAK!')

if bertaring and makan_daging or kepala_berbulu_lebat and nama:
    print("\nBenar.. Itu adalah SINGA JANTAN")
elif bertaring and makan_daging and warna_bulu_berpola and nama:
    print("\nBenar.. Itu adalah MACAN TUTUL")
elif makan_buah and berbulu_lebat and nama:
    print("\nBenar.. Itu adalah ORANGUTAN")
elif makan_buah and berleher_panjang and berbulu_tipis and warna_bulu_berpola and nama:
    print("\nBenar.. Itu adalah JERAPAH")
elif berbulu_lebat and bertaring and warna_bulu_berpola and makan_buah and nama:
    print("\nBenar.. Itu adalah PANDA")
elif berbulu_lebat and bertaring and warna_bulu_berpola and makan_buah and nama:
    print("\nBenar.. Itu adalah BERUANG")
else:
    print("\nTidak ditemukan!")