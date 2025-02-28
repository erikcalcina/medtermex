TRUE_ENTS = [
    {"text": "John Doe", "label": "PERSON"},
    {"text": "Los Angeles", "label": "LOCATION"},
    {"text": "US", "label": "COUNTRY"},
]


TEST_CASES_EXACT = [
    ([TRUE_ENTS], [[]], 0.0, 0.0, 0.0),
    (
        [TRUE_ENTS],
        [[{"text": "John Doe", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "John Doe", "label": "PERSON"},
                {"text": "Los Angeles", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "John Doe", "label": "PERSON"},
                {"text": "Los Angeles", "label": "LOCATION"},
                {"text": "US", "label": "COUNTRY"},
            ]
        ],
        1.0,
        1.0,
        1.0,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "John", "label": "PERSON"}]],
        0.0,
        0.0,
        0.0,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "John Doe", "label": "LOCATION"}]],
        0.0,
        0.0,
        0.0,
    ),
]

TEST_CASES_RELAXED = [
    ([TRUE_ENTS], [[]], 0.0, 0.0, 0.0),
    (
        [TRUE_ENTS],
        [[{"text": "John Doe", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "John Doe", "label": "PERSON"},
                {"text": "Los Angeles", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "John Doe", "label": "PERSON"},
                {"text": "Los Angeles", "label": "LOCATION"},
                {"text": "US", "label": "COUNTRY"},
            ]
        ],
        1.0,
        1.0,
        1.0,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "John Doe Jr.", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "John Doe", "label": "PERSON"},
                {"text": "Los Angeles, City of Angels", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
]


TEST_CASES_OVERLAP = [
    ([TRUE_ENTS], [[]], 0.0, 0.0, 0.0),
    (
        [TRUE_ENTS],
        [[{"text": "John Doe", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "John Doe", "label": "PERSON"},
                {"text": "Los Angeles", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "John Doe", "label": "PERSON"},
                {"text": "Los Angeles", "label": "LOCATION"},
                {"text": "US", "label": "COUNTRY"},
            ]
        ],
        1.0,
        1.0,
        1.0,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "John", "label": "PERSON"}]],
        4 / 8,
        4 / 24,
        4 / 16,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "John", "label": "LOCATION"}]],
        1 / 11,
        1 / 33,
        1 / 22,
    ),
]
