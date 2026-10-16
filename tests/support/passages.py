"""Real passages for the real-model tests: three authors on three different subjects."""

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
