"""Published records must not identify the Mac or its user (2026-11-20): the per-user temporary directory, the
account name (the owner in `ls -l`), host names and network addresses had reached the published traces and logs.
The public handle in github.com and huggingface.co links stays, and workspace paths keep the shape a replay needs."""

import unittest

from solvetime import publish, validate

HOME = "/Users/alice"
NAMES = sorted([("ab/k3j9x2_q8w7e6r5t4y3u2i1o0p9a0000gn", "xx/user"), ("k3j9x2_q8w7e6r5t4y3u2i1o0p9a0000gn", "user"),
                ("Alices-Mac-Mini", "mac"), ("Alice's Mac Mini", "mac"), ("Alice Example", "user"),
                ("192.168.1.2", "192.0.2.1"), ("2408:8207:abcd::1", "2001:db8::1")], key=lambda p: -len(p[0]))


TEMP = ("ab", "k3j9x2_q8w7e6r5t4y3u2i1o0p9a0000gn")


def scrub(text):
    return publish.scrub(text, names=NAMES, home=HOME, account="alice", temp=TEMP)


class Scrub(unittest.TestCase):
    def test_home_and_workspace(self):
        path = "/private/var/folders/ab/k3j9x2_q8w7e6r5t4y3u2i1o0p9a0000gn/T/solvetime-abc123/work/app.py"
        out = scrub(f"cd {path} && cat {HOME}/notes.txt")
        self.assertEqual(out, "cd /private/var/folders/xx/user/T/solvetime-abc123/work/app.py && cat ~/notes.txt")
        self.assertRegex(out, validate.WORK)

    def test_account_name(self):
        self.assertEqual(scrub("drwxr-xr-x@ 3 alice  staff  96 Nov 20 12:00 ."), "drwxr-xr-x@ 3 user  staff  96 Nov 20 12:00 .")
        for link in ("https://github.com/alice/solvetime/pull/46", "https://huggingface.co/alice/raceline-qwen3.8-27b"):
            self.assertEqual(scrub(link), link)
        self.assertEqual(scrub("alicewonder and malice"), "alicewonder and malice")

    def test_account_name_in_saved_traces(self):
        # Each of these reached a published trace under the rule before (2026-11-22).
        self.assertEqual(scrub(r"Nov 19 12:28 work\nalice\ndrwxr-xr-x@"), r"Nov 19 12:28 work\nuser\ndrwxr-xr-x@")
        self.assertEqual(scrub(r"work\\nalice\\ndrwx"), r"work\\nuser\\ndrwx")   # escaped twice, in a tool message
        self.assertEqual(scrub("/private/tmp/claude-501/-Users-alice-project-model/4a97"),
                         "/private/tmp/claude-501/-Users-user-project-model/4a97")
        self.assertEqual(scrub(r"96 Nov 15 19:03 claude-mcp-browser-bridge-alice\ndrwxrwxrwx"),
                         r"96 Nov 15 19:03 claude-mcp-browser-bridge-user\ndrwxrwxrwx")
        self.assertEqual(scrub("alice_backup and alice.local"), "user_backup and user.local")
        # still kept: a longer word, and the public handle
        self.assertEqual(scrub(r"\nmalice and unalice"), r"\nmalice and unalice")
        for public in ("https://hf.co/alice/x", "thanks @alice", "github.com/alice"):
            self.assertEqual(scrub(public), public)

    def test_host_names_full_name_and_addresses(self):
        line = "INFO - Auto-detected server aliases: ['localhost', '127.0.0.1', 'Alices-Mac-Mini.local', '192.168.1.2']"
        out = scrub(line)
        self.assertNotIn("Alices", out)
        self.assertNotIn("192.168.1.2", out)
        self.assertEqual(scrub("Alices-Mac-Mini.local and Alice's Mac Mini, by Alice Example"), "mac.local and mac, by user")
        self.assertEqual(scrub("on a 24 GB Mac mini"), "on a 24 GB Mac mini")
        self.assertEqual(scrub("http://192.168.1.2:8080/ and 192.168.1.25 and 10.192.168.1.2"),
                         "http://192.0.2.1:8080/ and 192.168.1.25 and 10.192.168.1.2")
        self.assertEqual(scrub("[2408:8207:abcd::1]:443 and 2408:8207:abcd::12"), "[2001:db8::1]:443 and 2408:8207:abcd::12")

    def test_the_id_typed_from_memory(self):
        # cut short, or misspelled after its first characters, as agents typed it (2026-11-20)
        self.assertEqual(scrub("cat /private/var/folders/ab/k3j9x2_q8w7e6r5/T/solvetime-xyz/work/a.py"),
                         "cat /private/var/folders/xx/user/T/solvetime-xyz/work/a.py")
        self.assertEqual(scrub("ls /var/folders/ab/k3j9x2_q8wzzzz/T"), "ls /var/folders/xx/user/T")
        self.assertEqual(scrub("ls /var/folders/ab/"), "ls /var/folders/xx/")
        system = "/var/folders/zz/zyxvpxvq6csfxvn_n0000000000000/T"
        self.assertEqual(scrub(system), system)
        self.assertRegex(scrub("/var/folders/ab/k3j9x2_q8w7e6/T/solvetime-xyz/work"), validate.WORK)

    def test_scrubbing_twice_changes_nothing_more(self):
        text = f"{HOME}/x alice  staff Alices-Mac-Mini 192.168.1.2 /var/folders/ab/k3j9x2_q8w7e6r5t4y3u2i1o0p9a0000gn/T"
        self.assertEqual(scrub(scrub(text)), scrub(text))


if __name__ == "__main__":
    unittest.main()
