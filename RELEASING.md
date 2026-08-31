# Releasing numerai-tools

numerai-tools ships to **PyPI only**. There are no container images, no ECR, and
no branch-triggered deploys — if you are thinking of the `master` → prod /
`staging` → staging image flow from our other repos, that does not apply here.
Users get everything by running `pip install numerai-tools`.

This flow is deliberately close to the ones in `numerapi/RELEASING.md` and
`numerai-cli/RELEASING.md`, so switching between the three repos should not
require re-learning anything. The differences that matter are called out under
[Caveats specific to this repo](#caveats-specific-to-this-repo).

## The model

Two rules explain everything else:

1. **`version` under `[project]` in `pyproject.toml` is the release.** The git
   tag is only the trigger. Whatever that string says is what lands on PyPI.
2. **The version string picks the channel**, not the branch. A
   [PEP 440](https://peps.python.org/pep-0440/) pre-release (`0.7.0.dev0`) is
   invisible to `pip install numerai-tools`; a final version (`0.7.0`) is what
   everyone gets by default.

| Ref | Role |
| --- | --- |
| `<user>/<topic>` | all work; branch off `master` |
| `master` | released state; **final** releases are cut here |
| `X.Y.Z.devN` tag | publishes a pre-release, from **any** branch |
| `X.Y.Z` tag | publishes a final release, **must** be on `master` |

**Merging does not publish.** A push to `master` runs the test matrix and
nothing else. Only pushing a tag publishes.

| What a user runs | What they get |
| --- | --- |
| `pip install numerai-tools` / `pip install -U numerai-tools` | latest **final** version |
| `pip install 'numerai-tools==0.7.0.dev0'` | that exact pre-release |
| `pip install --pre numerai-tools` | latest including pre-releases |

## Conventions

- **Tags are bare version numbers and must match `pyproject.toml` exactly:**
  `0.7.0`, `0.7.0.dev0`. A `v` prefix is not allowed — CI rejects `v0.7.0` with
  an explicit error.
- Use the canonical PEP 440 spelling with the dot: `0.7.0.dev0`, not
  `0.7.0dev0`. Both normalize to the same release, but the canonical form avoids
  confusion.
- Pre-releases use `.devN`. Increment `N` for each beta on the same version line.
- **A version number can never be reused.** PyPI permanently rejects
  re-uploading a version, even one that was deleted. If you burn a number, move
  to the next.

## Develop without releasing

```bash
git checkout master && git pull
git checkout -b josh/some-feature
# ... work ...
git push -u origin josh/some-feature
gh pr create --base master
```

The Python 3.11–3.14 × pandas 2.2/2.3/3.0 matrix, ruff, and mypy run on every
push. No tag means nothing is published, on any branch including `master`. Leave
`pyproject.toml` alone until you are actually cutting something.

## Cut a beta

For beta users who need the code before it is stable. There is no integration
branch here — cut it straight from your topic branch, before it merges.

```bash
git checkout josh/some-feature

# pyproject.toml:  version = "0.7.0.dev0"
git commit -am "numerai-tools 0.7.0.dev0"
git push origin josh/some-feature     # publishes nothing

# tag and push — this is the release event
git tag 0.7.0.dev0
git push origin 0.7.0.dev0
```

Verify:

```bash
gh run list --workflow=test-and-deploy.yml --limit 1   # expect success
pip install 'numerai-tools==0.7.0.dev0'                # what beta users run
pip install -U numerai-tools                           # must NOT be the dev version
```

Tell beta users to install the exact version. Note that `pip index versions` and
the simple index can lag a few minutes behind a successful publish on CDN cache;
an exact-version install works immediately.

For the next beta, repeat with `.dev1`, `.dev2`, …

To try a beta against the monorepo, pin the exact version in the relevant
service's `pyproject.toml` on a branch — never merge a `.devN` pin to the
monorepo's `master`.

## Promote to a final release (from `master`)

Flip the version to final **as the last commit before merging**, so `master`
never holds a pre-release string and picks up the release version atomically at
merge.

```bash
git checkout josh/some-feature

# pyproject.toml:  version = "0.7.0"      (drop the .devN suffix)
git commit -am "numerai-tools 0.7.0"
git push origin josh/some-feature

gh pr create --base master --title "numerai-tools 0.7.0"
gh pr merge <n> --squash                 # publishes nothing

git checkout master && git pull
grep '^version' pyproject.toml           # must read exactly 0.7.0
git tag 0.7.0
git push origin 0.7.0
```

Verify with `pip install -U numerai-tools` in a clean virtualenv.

Then bump the numerai-tools pin in `tournament-monorepo` — it is pinned in
`init-round`, `v2-submissions-cryptosignals`, `crypto-data`, and `tests`. The
pins are caret constraints (`^0.5.3`), so a minor bump does **not** get picked up
automatically and needs an explicit edit plus `poetry lock`. That PR moving
through the monorepo's own staging → master is what carries the new
numerai-tools into staging and prod images. Never pin a `.devN` version in
anything that reaches prod.

## Hotfix a released version

`master` is the only long-lived branch, so a hotfix is just the normal flow with
a patch bump:

```bash
git checkout -b hotfix/0.7.1 master
# fix + pyproject.toml 0.7.1
gh pr create --base master
# after merge:
git checkout master && git pull
git tag 0.7.1 && git push origin 0.7.1
```

## What CI enforces

`.github/workflows/test-and-deploy.yml` runs on every branch push and on tag
pushes that start with a digit (and on `v`-prefixed tags, solely to reject
them). On a tag push it refuses to publish unless:

1. **The tag has no `v` prefix.** `v0.7.0` fails with an error telling you to
   re-tag as `0.7.0`.
2. **The tag matches `pyproject.toml`.** Compared as normalized PEP 440
   versions, so `0.7.0dev0` and `0.7.0.dev0` are equivalent, but `0.7.0` against
   a `pyproject.toml` of `0.7.0.dev0` fails.
3. **Final releases point at a commit on `master`.** Pre-releases skip this
   check, so betas can be cut from a topic branch but a final one cannot.
4. **The full test matrix passes on the tagged commit.** See the first caveat
   below.

## Caveats specific to this repo

**Tests gate the publish here — unlike numerapi and numerai-cli.** Those two put
the publish in a separate `pypi.yml`, and GitHub Actions cannot express a
cross-workflow dependency, so in those repos a red suite will not stop an
upload. Here the `deploy` job lives in the same workflow and `needs` both
`verify-tag` and `test-numerai-tools`, so the matrix re-runs against the tagged
commit and a failure blocks the release. Two consequences: a tag push costs a
few extra minutes before anything is published, and a flaky test can block a
release you already tagged. If that happens, **re-run the failed job — do not
delete and re-push the tag**, and never bump the version to work around it.

**There is no `preview` branch, and that is on purpose.** numerapi has one
because `tournament-monorepo` pins it and needs somewhere unreleased work can
sit indefinitely. `tournament-monorepo` pins numerai-tools too, but that is not
a reason to add one: since nothing publishes on a merge, `master` can hold
unreleased work safely and the monorepo simply stays on its existing pin until
someone bumps it. A second long-lived branch would be pure overhead. The
practical consequence is that betas are cut from topic branches rather than from
an integration branch.

**Merging used to publish, and that is why the history looks the way it does.**
Until this flow, the `deploy` job fired on every push to `master`, so the version
bump in a PR *was* the release. That coupling meant a README fix could not land
without either cutting a version or turning `master` red on a duplicate-version
error. It also meant there was no CI path to a pre-release at all: all 21
`.devN` versions on PyPI (`0.4.0.dev0`–`0.6.0.dev0`) were published by hand with
a local `poetry publish` and a personal token, ungated by tests and untraceable
to a commit. Don't do that anymore — tag instead.

**The repo has no tags yet.** `0.6.0` and everything before it were published by
branch-push CI or by hand, and none of them have a corresponding tag. The tag
history starts fresh with the first release cut under this flow. **Do not
retro-tag old releases** — any new tag starting with a digit triggers a publish
attempt that will fail on a duplicate version.

**Version is only in `pyproject.toml`.** The package exposes no `__version__`
and nothing else in the tree hardcodes it, so `[project] version` is the single
place to edit. CI reads it with `tomllib`, not a regex.

**There is no CHANGELOG.md.** numerapi's flow includes a changelog edit at each
version bump; this repo has never had one, and the release notes live in the PR
title and `README.md`. Adding one is optional and not assumed anywhere above.

**The PyPI secret is `PYPI_API_KEY`**, as in numerai-cli, not `PYPI_API_TOKEN`
as in numerapi. Publishing uses `poetry publish --build` — it is the path this
repo's token is already set up for.

**Pre-releases are not new here.** PyPI already holds 21 of them from the 0.4.x
through 0.6.x lines. The `.devN` convention above matches what is already there.

## Troubleshooting

**`File already exists` on publish.** That version is already on PyPI. Bump to
the next number — you cannot re-upload, and you cannot fix it by deleting the
release on PyPI either.

**Tag mismatch error.** You tagged without bumping `pyproject.toml`, or vice
versa. Fix `pyproject.toml`, commit, delete the tag locally and on origin
(`git push origin :refs/tags/X.Y.Z`), then re-tag. Deleting a tag never
publishes anything.

**"Release tags must be bare version numbers."** You tagged `v0.7.0` out of
habit. Delete the tag locally and on origin, then re-tag as `0.7.0`.

**"Final release tags must point at a commit on master."** You tagged a
suffix-free version on a topic branch. Either merge to `master` first, or cut it
as a `.devN` pre-release instead.

**The publish job was skipped.** `deploy` only runs on tag pushes. If you pushed
a branch, that is working as intended — nothing publishes on a merge.

**A bad version is already public.** You cannot unpublish, but you can `yank` it
on PyPI, which hides it from resolution while leaving existing pins working.
Then ship the fix as the next version.
