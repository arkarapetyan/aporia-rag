"""Real passages for the real-model tests: Kant, Hobbes and Mill, each on a subject of their own."""

from support.builders import make_chunk

# Abbott's Kant, Hobbes in his 1651 spelling, and Mill.
KANT = make_chunk(
    "kant/first-section#0",
    "Nothing can possibly be conceived in the world, or even out of it, which can be called "
    "good, without qualification, except a Good Will.",
    author="Immanuel Kant",
    title="Fundamental Principles of the Metaphysic of Morals",
    section="First Section",
)
HOBBES = make_chunk(
    "hobbes/chapter-xiii#0",
    "Hereby it is manifest, that during the time men live without a common Power to keep them "
    "all in awe, they are in that condition which is called Warre; and such a warre, as is of "
    "every man, against every man.",
    author="Thomas Hobbes",
    title="Leviathan",
    section="Chapter XIII",
)
MILL = make_chunk(
    "mill/chapter-ii#0",
    "The creed which accepts as the foundation of morals, Utility, or the Greatest Happiness "
    "Principle, holds that actions are right in proportion as they tend to promote happiness.",
    author="John Stuart Mill",
    title="Utilitarianism",
    section="Chapter II",
)

# The two passages `ask` graded for "How do Kant and Mill differ on lying?" on 2026-10-20: each
# author's best match for the question. Kant's is on lying; Mill's is not.
KANT_ON_LYING = make_chunk(
    "kant-metaphysic-of-morals/first-section#28",
    "Now it is a wholly different thing to be truthful from duty and to be so from apprehension "
    "of injurious consequences. In the first case, the very notion of the action already implies "
    "a law for me; in the second case, I must first look about elsewhere to see what results may "
    "be combined with it which would affect myself. For to deviate from the principle of duty is "
    "beyond all doubt wicked; but to be unfaithful to my maxim of prudence may often be very "
    "advantageous to me, although to abide by it is certainly safer. The shortest way, however, "
    "and an unerring one, to discover the answer to this question whether a lying promise is "
    'consistent with duty, is to ask myself, "Should I be content that my maxim (to extricate '
    "myself from difficulty by a false promise) should hold good as a universal law, for myself as "
    'well as for others?"',
    author="Immanuel Kant",
    title="Fundamental Principles of the Metaphysic of Morals",
    section="First Section",
)
MILL_ON_DISSENT = make_chunk(
    "mill-on-liberty/chapter-ii#95",
    "When there are persons to be found, who form an exception to the apparent unanimity of the "
    "world on any subject, even if the world is in the right, it is always probable that "
    "dissentients have something worth hearing to say for themselves, and that truth would lose "
    "something by their silence.",
    author="John Stuart Mill",
    title="On Liberty",
    section="Chapter II",
)
# Mill's own passage on lying (Utilitarianism II), which the comparison wording missed; the
# comparison spike of 2026-10-20 graded these three against each author's own question.
MILL_ON_LYING = make_chunk(
    "mill-utilitarianism/chapter-ii#54",
    "we feel that the violation, for a present advantage, of a rule of such transcendent "
    "expediency, is not expedient, and that he who, for the sake of a convenience to himself or "
    "to some other individual, does what depends on him to deprive mankind of the good, and "
    "inflict upon them the evil, involved in the greater or less reliance which they can place in "
    "each other's word, acts the part of one of their worst enemies. Yet that even this rule, "
    "sacred as it is, admits of possible exceptions, is acknowledged by all moralists; the chief "
    "of which is when the withholding of some fact (as of information from a male-factor, or of "
    "bad news from a person dangerously ill) would preserve some one (especially a person other "
    "than oneself) from great and unmerited evil, and when the withholding can only be effected "
    "by denial.",
    author="John Stuart Mill",
    title="Utilitarianism",
    section="Chapter II",
)
