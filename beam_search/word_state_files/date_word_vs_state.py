from beam_search.GuidedBeamSearch import STARTING_STATE

DATE_STATES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

DATE_STATE_TRANSITION = {
    "A": [
        "C",
        "F"
    ],
    "B": [
        "C",
        "D",
        "F"
    ],
    "C": [
        "F"
    ],
    "D": [
        "F"
    ],
    "E": [
        "F"
    ],
    "F": [
        "G",
        "H"
    ],
    "G": [
        "I"
    ],
    "H": [
        "J"
    ],
    "I": [
        "J"
    ],
    "J": [
        "K"
    ],
    STARTING_STATE: [
        "A",
        "B",
        "C",
        "D",
        "E"
    ],
    "terminal_states": [
        "F",
        "K"
    ]
}
DATE_STATES_VS_WORDS = {
    "A": [
        "thirty"
    ],
    "B": [
        "twenty"
    ],
    "C": [
        "one",
        "first"
    ],
    "D": [
        "two",
        "second",
        "three",
        "third",
        "four",
        "fourth",
        "five",
        "fifth",
        "six",
        "sixth",
        "seven",
        "seventh",
        "eight",
        "eighth",
        "nine",
        "ninth"
    ],
    "E": [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
        "thirtieth"
    ],
    "F": [
        "january",
        "jan",
        "february",
        "feb",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december"
    ],
    "G": [
        "two"
    ],
    "H": [
        "twenty"
    ],
    "I": [
        "thousand"
    ],
    "J": [
        "twenty"
    ],
    "K": [
        "one",
        "two",
        "three"
    ]
}

