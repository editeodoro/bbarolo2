"""
Sphinx extension of the BBarolo documentation: the parameters of a task, from the keyword table of the
code (bbarolo/keywords.py), so that the pages always list every keyword with its default.

    ```{bbparams} 3DFIT
    ```

renders the keywords of the task 3DFIT (ALL: the general parameters, used by every task), each with its
aliases, type, default and description, in the order of the table. Every keyword gets a target,
par-<task>-<keyword> in lower case (e.g. par-3dfit-free), for references such as
{ref}`FREE <par-3dfit-free>`. The description is Keyword.doc (Markdown) if the keyword has one, else
its one-line help.
"""

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

_TYPES = {"str": "text", "lstr": "text", "ustr": "text", "pos": "text", "flag": "true or false",
          "int": "integer", "float": "number", "cdens": "number", "flagorint": "integer (or true/false)",
          "ivec": "list of integers", "fvec": "list of numbers", "fvec2": "list of numbers",
          "fvec3": "list of numbers", "ivec6": "list of integers"}


def _keywords(task):
    from bbarolo import keywords as K
    if task == "ALL":
        return [k for k in K.KEYWORDS if k.tasks[0] == "ALL"]
    return K.keywords_for(task)


def _default(kw, task):
    from bbarolo import keywords as K
    from bbarolo.params import _default_param
    v = K.TASK_DEFAULTS.get(task, {}).get(kw.name, kw.value(_default_param()))
    s = K._fmtval(v)
    return None if s in ("-1", '""', "") else s


def _type(kw):
    from bbarolo.params import _RING_KEYS
    if kw.name in _RING_KEYS:
        return "a number, one number per ring, or file(name,column)"
    return _TYPES.get(kw.kind, kw.kind)


class BBParams(SphinxDirective):
    required_arguments = 1
    has_content = False

    def run(self):
        task = self.arguments[0].strip().upper()
        kws = _keywords(task)
        if not kws:
            raise self.error(f"bbparams: no keywords for task {task!r}")
        out = nodes.definition_list(classes=["bbparams"])
        for kw in kws:
            item = nodes.definition_list_item()
            tid = nodes.make_id(f"par-{task}-{kw.name}".lower())
            term = nodes.term()
            term += nodes.literal(text=kw.name)
            if kw.aliases:
                term += nodes.inline(text="  (also " + ", ".join(kw.aliases) + ")", classes=["bbparams-aliases"])
            term["ids"].append(tid)
            self.env.get_domain("std").note_hyperlink_target(tid, self.env.docname, tid, kw.name)
            item += term
            body = nodes.definition()
            text = (getattr(kw, "doc", None) or kw.help.rstrip(".") + ".").strip()
            meta = [f"*Type:* {_type(kw)}"]
            d = _default(kw, task)
            meta.append(f"*Default:* `{d}`" if d is not None else "*Default:* not set")
            lines = text.splitlines() + ["", " · ".join(meta)]
            self.state.nested_parse(StringList(lines, source=f"keyword {kw.name}"), self.content_offset, body)
            item += body
            out += item
        return [out]


def setup(app):
    app.add_directive("bbparams", BBParams)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
