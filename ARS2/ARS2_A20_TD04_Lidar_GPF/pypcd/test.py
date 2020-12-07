if __name__ == "__main__":
    line = 'joker JOKER-4 4 4 4'
    p1, p2 = line.split('-')
    if p1 == 'joker JOKER' or p2 == 'joker JOKER':
        print('joker JOKER')
    else:
        p1 = p1.split()
        p2 = p2.split()
        all_card = "3 4 5 6 7 8 9 10 J Q K A 2 joker JOKER"
        if len(p1) == len(p2):
            if all_card.index(p1[0]) > all_card.index(p2[0]):
                print(p1)
            else:
                print(p2)
        else:
            if len(p1) == 4:
                print(p1)
            elif len(p2) == 4:
                print(p2)
            else:
                print('ERROR')