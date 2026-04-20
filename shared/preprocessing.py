import os
import glob
import xml.etree.ElementTree as ET

LABEL2ID = {"negative": 0, "mechanism": 1, "effect": 2, "advise": 3, "int": 4}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

CORPUS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DDICorpus")
TRAIN_DIR   = os.path.join(CORPUS_ROOT, "Train")
TEST_DIR    = os.path.join(CORPUS_ROOT, "Test", "Test for DDI Extraction task")


def load_xml_files(folder_path):
    """
    Parse all DDI 2013 XML files under folder_path (or a list of paths).
    Returns raw example dicts with character offsets preserved so each
    track can do its own transformation (entity marking, graph building, etc).

    Each dict:
        text     : raw sentence string
        e1_start : char offset start of drug 1 (inclusive)
        e1_end   : char offset end   of drug 1 (inclusive)
        e1_text  : drug 1 surface form
        e2_start : char offset start of drug 2
        e2_end   : char offset end   of drug 2
        e2_text  : drug 2 surface form
        label    : int 0-4
    """
    if isinstance(folder_path, list):
        examples = []
        for path in folder_path:
            examples.extend(_parse_folder(path))
        return examples
    return _parse_folder(folder_path)


def _parse_folder(folder_path):
    xml_files = sorted(glob.glob(
        os.path.join(folder_path, "**", "*.xml"), recursive=True
    ))
    examples = []
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for sentence in root.iter("sentence"):
            sent_text = sentence.attrib.get("text", "").strip()
            if not sent_text:
                continue

            entities = {}
            for ent in sentence.findall("entity"):
                eid    = ent.attrib["id"]
                offset = ent.attrib["charOffset"].split(";")[0]
                start, end = map(int, offset.split("-"))
                etext  = ent.attrib.get("text", sent_text[start:end + 1])
                entities[eid] = (start, end, etext)

            for pair in sentence.findall("pair"):
                e1_id = pair.attrib["e1"]
                e2_id = pair.attrib["e2"]
                if e1_id not in entities or e2_id not in entities:
                    continue
                ddi   = pair.attrib.get("ddi", "false").lower() == "true"
                itype = pair.attrib.get("type", "").lower() if ddi else "negative"
                label = LABEL2ID.get(itype, 0)

                e1_start, e1_end, e1_text = entities[e1_id]
                e2_start, e2_end, e2_text = entities[e2_id]
                examples.append({
                    "text":     sent_text,
                    "e1_start": e1_start,
                    "e1_end":   e1_end,
                    "e1_text":  e1_text,
                    "e2_start": e2_start,
                    "e2_end":   e2_end,
                    "e2_text":  e2_text,
                    "label":    label,
                })
    return examples
