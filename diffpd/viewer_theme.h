#pragma once
// Color themes of the viewer. To add a theme: append one Palette to kPalettes[] (it appears in the F2 cycle).
// Only roles are read by the drawing code (theme().text, theme().orange, ...), never palette slots.
#include "raylib.h"

#include <cstdint>

struct Palette
{
    const char* name;
    Color       ramp[8]; // neutrals from the background [0] (darkest) to the strongest foreground [7] (lightest)
    Color       yellow, orange, red, magenta, blue, cyan, green;
    Color       lime;    // bright green for guess particles in contact: must stand out on the guess cloth
};

namespace theme_detail
{
constexpr Color rgb(std::uint32_t hex) { return { (unsigned char)(hex >> 16), (unsigned char)(hex >> 8), (unsigned char)hex, 255 }; }
} // namespace theme_detail

inline const Palette kPalettes[] = {
    { "Solarized Dark",
      { theme_detail::rgb(0x002b36), theme_detail::rgb(0x073642), theme_detail::rgb(0x586e75), theme_detail::rgb(0x657b83),
        theme_detail::rgb(0x839496), theme_detail::rgb(0x93a1a1), theme_detail::rgb(0xeee8d5), theme_detail::rgb(0xfdf6e3) },
      theme_detail::rgb(0xb58900), theme_detail::rgb(0xcb4b16), theme_detail::rgb(0xdc322f), theme_detail::rgb(0xd33682),
      theme_detail::rgb(0x268bd2), theme_detail::rgb(0x2aa198), theme_detail::rgb(0x859900), theme_detail::rgb(0xb0e635) },
    { "Gruvbox Dark",
      { theme_detail::rgb(0x282828), theme_detail::rgb(0x3c3836), theme_detail::rgb(0x665c54), theme_detail::rgb(0x7c6f64),
        theme_detail::rgb(0xa89984), theme_detail::rgb(0xd5c4a1), theme_detail::rgb(0xebdbb2), theme_detail::rgb(0xfbf1c7) },
      theme_detail::rgb(0xfabd2f), theme_detail::rgb(0xfe8019), theme_detail::rgb(0xfb4934), theme_detail::rgb(0xd3869b),
      theme_detail::rgb(0x83a598), theme_detail::rgb(0x8ec07c), theme_detail::rgb(0xb8bb26), theme_detail::rgb(0xd4e030) },
    { "Nord",
      { theme_detail::rgb(0x2e3440), theme_detail::rgb(0x3b4252), theme_detail::rgb(0x4c566a), theme_detail::rgb(0x616e88),
        theme_detail::rgb(0xa0aabf), theme_detail::rgb(0xd8dee9), theme_detail::rgb(0xe5e9f0), theme_detail::rgb(0xeceff4) },
      theme_detail::rgb(0xebcb8b), theme_detail::rgb(0xd08770), theme_detail::rgb(0xbf616a), theme_detail::rgb(0xb48ead),
      theme_detail::rgb(0x81a1c1), theme_detail::rgb(0x88c0d0), theme_detail::rgb(0xa3be8c), theme_detail::rgb(0xb2e07a) },
    { "Catppuccin Mocha",
      { theme_detail::rgb(0x1e1e2e), theme_detail::rgb(0x313244), theme_detail::rgb(0x585b70), theme_detail::rgb(0x6c7086),
        theme_detail::rgb(0x9399b2), theme_detail::rgb(0xbac2de), theme_detail::rgb(0xcdd6f4), theme_detail::rgb(0xf5e0dc) },
      theme_detail::rgb(0xf9e2af), theme_detail::rgb(0xfab387), theme_detail::rgb(0xf38ba8), theme_detail::rgb(0xcba6f7),
      theme_detail::rgb(0x89b4fa), theme_detail::rgb(0x94e2d5), theme_detail::rgb(0xa6e3a1), theme_detail::rgb(0x8ce04f) },
    { "Dracula",
      { theme_detail::rgb(0x282a36), theme_detail::rgb(0x343746), theme_detail::rgb(0x44475a), theme_detail::rgb(0x6272a4),
        theme_detail::rgb(0x9ea8cc), theme_detail::rgb(0xd6d8e8), theme_detail::rgb(0xe8e8e2), theme_detail::rgb(0xf8f8f2) },
      theme_detail::rgb(0xf1fa8c), theme_detail::rgb(0xffb86c), theme_detail::rgb(0xff5555), theme_detail::rgb(0xff79c6),
      theme_detail::rgb(0xbd93f9), theme_detail::rgb(0x8be9fd), theme_detail::rgb(0x50fa7b), theme_detail::rgb(0xa5f23a) },
    { "Tokyo Night",
      { theme_detail::rgb(0x1a1b26), theme_detail::rgb(0x24283b), theme_detail::rgb(0x414868), theme_detail::rgb(0x565f89),
        theme_detail::rgb(0x9aa5ce), theme_detail::rgb(0xa9b1d6), theme_detail::rgb(0xc0caf5), theme_detail::rgb(0xe0e6ff) },
      theme_detail::rgb(0xe0af68), theme_detail::rgb(0xff9e64), theme_detail::rgb(0xf7768e), theme_detail::rgb(0xbb9af7),
      theme_detail::rgb(0x7aa2f7), theme_detail::rgb(0x7dcfff), theme_detail::rgb(0x9ece6a), theme_detail::rgb(0xbaf26b) },
    { "One Dark",
      { theme_detail::rgb(0x282c34), theme_detail::rgb(0x353b45), theme_detail::rgb(0x4b5263), theme_detail::rgb(0x5c6370),
        theme_detail::rgb(0x9aa1ae), theme_detail::rgb(0xabb2bf), theme_detail::rgb(0xc8ccd4), theme_detail::rgb(0xe9ebef) },
      theme_detail::rgb(0xe5c07b), theme_detail::rgb(0xd19a66), theme_detail::rgb(0xe06c75), theme_detail::rgb(0xc678dd),
      theme_detail::rgb(0x61afef), theme_detail::rgb(0x56b6c2), theme_detail::rgb(0x98c379), theme_detail::rgb(0xb4e86e) },
    { "Monokai",
      { theme_detail::rgb(0x272822), theme_detail::rgb(0x3e3d32), theme_detail::rgb(0x57574a), theme_detail::rgb(0x75715e),
        theme_detail::rgb(0xa59f85), theme_detail::rgb(0xcfcfc2), theme_detail::rgb(0xf8f8f2), theme_detail::rgb(0xffffff) },
      theme_detail::rgb(0xe6db74), theme_detail::rgb(0xfd971f), theme_detail::rgb(0xf92672), theme_detail::rgb(0xae81ff),
      theme_detail::rgb(0x5a8dee), theme_detail::rgb(0x66d9ef), theme_detail::rgb(0xa6e22e), theme_detail::rgb(0xbff03a) },
};

constexpr int kPaletteCount = (int)(sizeof(kPalettes) / sizeof(kPalettes[0]));

// The roles every drawing call reads, derived from one palette by make_theme().
struct Theme
{
    Color background, panel, panel_fill, border, text_dim, text, text_hi;
    Color guess, collider, collider_hi;
    Color yellow, orange, red, magenta, blue, cyan, green, lime;
    Color red_text, orange_text; // accents pulled toward text_hi: the pure accents are too dim/bright for small text
    Color axis[3];               // X, Y, Z
    Color seam[4];               // sewing links by seam level 1..4
};

inline Theme make_theme(const Palette& p)
{
    Theme t;
    t.background  = p.ramp[0];
    t.panel       = p.ramp[1];
    t.panel_fill  = { p.ramp[1].r, p.ramp[1].g, p.ramp[1].b, 215 }; // translucent over the 3D view
    t.border      = p.ramp[2];
    t.text_dim    = p.ramp[4];
    t.text        = p.ramp[5];
    t.text_hi     = p.ramp[6];
    t.collider    = p.ramp[5];
    t.collider_hi = p.ramp[6];
    t.guess       = p.ramp[7];
    t.yellow = p.yellow; t.orange = p.orange; t.red = p.red; t.magenta = p.magenta;
    t.blue = p.blue; t.cyan = p.cyan; t.green = p.green; t.lime = p.lime;
    t.red_text    = ColorLerp(p.red,    p.ramp[6], 0.25f);
    t.orange_text = ColorLerp(p.orange, p.ramp[6], 0.25f);
    t.axis[0] = p.red; t.axis[1] = p.green; t.axis[2] = p.blue;
    t.seam[0] = p.yellow; t.seam[1] = p.cyan; t.seam[2] = p.magenta; t.seam[3] = p.green;
    return t;
}

inline Theme g_theme = make_theme(kPalettes[0]);
inline const Theme& theme() { return g_theme; }
