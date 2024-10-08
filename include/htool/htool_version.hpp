#ifndef HTOOL_VERSION_HPP
#define HTOOL_VERSION_HPP

#define HTOOL_VERSION_MAJOR 0
#define HTOOL_VERSION_MINOR 9
#define HTOOL_VERSION_SUBMINOR 0

#define HTOOL_VERSION_EQ(MAJOR, MINOR, SUBMINOR) \
    ((HTOOL_VERSION_MAJOR == (MAJOR)) && (HTOOL_VERSION_MINOR == (MINOR)) && (HTOOL_VERSION_SUBMINOR == (SUBMINOR)))

#define HTOOL_VERSION_ HTOOL_VERSION_EQ

#define HTOOL_VERSION_LT(MAJOR, MINOR, SUBMINOR) \
    (HTOOL_VERSION_MAJOR < (MAJOR) || (HTOOL_VERSION_MAJOR == (MAJOR) && (HTOOL_VERSION_MINOR < (MINOR) || (HTOOL_VERSION_MINOR == (MINOR) && (HTOOL_VERSION_SUBMINOR < (SUBMINOR))))))

#define HTOOL_VERSION_LE(MAJOR, MINOR, SUBMINOR) \
    (HTOOL_VERSION_LT(MAJOR, MINOR, SUBMINOR) || HTOOL_VERSION_EQ(MAJOR, MINOR, SUBMINOR))

#define HTOOL_VERSION_GT(MAJOR, MINOR, SUBMINOR) \
    (0 == HTOOL_VERSION_LE(MAJOR, MINOR, SUBMINOR))

#define HTOOL_VERSION_GE(MAJOR, MINOR, SUBMINOR) \
    (0 == HTOOL_VERSION_LT(MAJOR, MINOR, SUBMINOR))

#endif
