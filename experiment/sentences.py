"""The sentence sets used in the experiment.

Set A  english_kurdish   : English sentences about Kurdistan / Kurdish / Zakho / Duhok ...
Set B  english_other     : The SAME English sentence templates, but about other places
                           (Turkey, Germany, Japan ...). Using the same templates means the
                           only real difference between A and B is "Kurdish or not".
Set C  kurmanji          : Short everyday sentences written in Kurdish (Kurmanji, Latin script).
Set D  other_foreign     : Short everyday sentences in other languages written in Latin script
                           (Turkish, German, Spanish, ...). This is a control for set C, so we
                           can tell "Kurdish language" apart from simply "not English".
Plus small held-out test sets that are NOT used to build the direction, only to check it.
"""

TEMPLATES = [
    "Many people in {country} speak {lang}.",
    "{city} is a beautiful city in {country}.",
    "I visited {city} last summer with my family.",
    "The {lang} language has a long history.",
    "My grandmother tells old {lang} stories.",
    "The {people} are known for their hospitality.",
    "We ate traditional {lang} food in {city}.",
    "She is learning to write in {lang}.",
    "The mountains near {city} are full of snow in winter.",
    "He listened to {lang} music on the radio.",
    "The flag of {country} flies over the city hall.",
    "Students in {city} study {lang} at school.",
    "There is a new university in {city}.",
    "The history of {country} is complicated.",
    "Every spring, families in {city} have a picnic.",
    "A famous poet from {country} wrote this poem.",
    "The news from {country} was on television tonight.",
    "My friend from {city} invited me to a wedding.",
    "{lang} dancing is fun to watch at weddings.",
    "The road from {city} to the border was busy.",
]

KURDISH_ENTITY = dict(country="Kurdistan", lang="Kurdish", people="Kurds")
KURDISH_CITIES = ["Zakho", "Duhok", "Erbil", "Amedi", "Sulaymaniyah"]

OTHER_ENTITIES = [
    dict(country="Turkey", lang="Turkish", people="Turks", city="Istanbul"),
    dict(country="Germany", lang="German", people="Germans", city="Munich"),
    dict(country="Japan", lang="Japanese", people="Japanese", city="Kyoto"),
    dict(country="France", lang="French", people="French", city="Lyon"),
    dict(country="Spain", lang="Spanish", people="Spanish", city="Seville"),
    dict(country="Iran", lang="Persian", people="Iranians", city="Shiraz"),
    dict(country="Greece", lang="Greek", people="Greeks", city="Athens"),
    dict(country="Poland", lang="Polish", people="Poles", city="Krakow"),
    dict(country="Italy", lang="Italian", people="Italians", city="Naples"),
    dict(country="Russia", lang="Russian", people="Russians", city="Kazan"),
]


def english_kurdish():
    out = []
    for i, t in enumerate(TEMPLATES):
        e = dict(KURDISH_ENTITY, city=KURDISH_CITIES[i % len(KURDISH_CITIES)])
        out.append((i, t.format(**e)))
    return out


def english_other():
    out = []
    for i, t in enumerate(TEMPLATES):
        for k in range(2):  # two different countries per template
            e = OTHER_ENTITIES[(i + k * 5) % len(OTHER_ENTITIES)]
            out.append((i, t.format(**e)))
    return out


# Kurmanji (Latin script). Mostly everyday sentences WITHOUT place names, so the model
# has to recognise the language itself, not the word "Kurdistan".
KURMANJI = [
    "Navê min Azad e.",
    "Ez te gelek hez dikim.",
    "Tu çawa yî?",
    "Îro hewa gelek xweş e.",
    "Ez nanê germ dixwim.",
    "Dayika min li malê ye.",
    "Zarok li baxçeyê dilîzin.",
    "Çiya bilind û bedew in.",
    "Sibê ez ê biçim dibistanê.",
    "Bavê min mamoste ye.",
    "Av sar û paqij e.",
    "Me çay vexwar û em axivîn.",
    "Ev pirtûk gelek baş e.",
    "Li gundê me gelek dar hene.",
    "Rojbaş, tu bi xêr hatî.",
    "Ez birçî me, werin em xwarinê bixwin.",
    "Xwişka min nameyek nivîsî.",
    "Em ê êvarê herin bazarê.",
    "Ez ji Zaxoyê me.",
    "Em bi kurdî diaxivin.",
]

# Same kind of everyday sentences in other languages that also use the Latin alphabet.
OTHER_FOREIGN = [
    "Benim adım Ahmet.",                      # Turkish
    "Bugün hava çok güzel.",                  # Turkish
    "Annem evde.",                            # Turkish
    "Çocuklar bahçede oynuyor.",              # Turkish
    "Mein Name ist Anna.",                    # German
    "Heute ist das Wetter sehr schön.",       # German
    "Die Kinder spielen im Garten.",          # German
    "Mein Vater ist Lehrer.",                 # German
    "Me llamo Carlos.",                       # Spanish
    "Hoy hace muy buen tiempo.",              # Spanish
    "Mi madre está en casa.",                 # Spanish
    "Este libro es muy bueno.",               # Spanish
    "Je mange du pain chaud.",                # French
    "Les montagnes sont hautes et belles.",   # French
    "Nama saya Budi.",                        # Indonesian
    "Hari ini cuacanya sangat bagus.",        # Indonesian
    "Ik hou heel veel van jou.",              # Dutch
    "Het water is koud en schoon.",           # Dutch
    "Jak się masz?",                          # Polish
    "Mój ojciec jest nauczycielem.",          # Polish
]

# Held-out checks: different sentence shapes, never used to build the direction.
TEST_KURDISH = [
    "Newroz is the Kurdish new year, celebrated with fires on the hills.",
    "The Peshmerga protected the road to Duhok.",
    "In Zakho, the old Delal bridge crosses the Khabur river.",
    "Kurmanji and Sorani are two main Kurdish dialects.",
    "Kurdish families in Erbil drink tea after dinner.",
    "The Kurdish region of Iraq has its own parliament.",
    "A Kurdish singer from Duhok became famous on YouTube.",
    "Many Kurds live in Turkey, Iraq, Iran and Syria.",
]
TEST_OTHER = [
    "Diwali is the Hindu festival of lights, celebrated with lamps.",
    "The soldiers protected the road to Marseille.",
    "In Prague, the old Charles bridge crosses the Vltava river.",
    "Cantonese and Mandarin are two main Chinese languages.",
    "Turkish families in Izmir drink tea after dinner.",
    "The Basque region of Spain has its own parliament.",
    "A Greek singer from Patras became famous on YouTube.",
    "Many Armenians live in Russia, France and the United States.",
]
