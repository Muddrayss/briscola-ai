from briscola_ai.core.cards import Card, Suit, trick_winner

def check():
    br = Suit.BASTONI
    # stesso seme: A batte 3
    lead = Card(Suit.COPPE,"A"); foll = Card(Suit.COPPE,"3")
    assert trick_winner(lead,foll,br)==0
    # briscola batte non briscola
    lead = Card(Suit.COPPE,"K"); foll = Card(br,"2")
    assert trick_winner(lead,foll,br)==1
    # seme non seguito e senza briscola -> vince lead
    lead = Card(Suit.SPADE,"7"); foll = Card(Suit.COPPE,"A")
    assert trick_winner(lead,foll,br)==0
    print("OK: regole base")

if __name__=="__main__":
    check()
