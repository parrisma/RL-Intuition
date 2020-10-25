import yaml
import socket
from typing import Dict, List
from datetime import datetime
from src.lib.transformer import Transformer


class Settings:
    _header = "header"
    _version_tag = 'version'
    _date_tag = 'date'
    _description_tag = 'description'
    _header_items = [[_version_tag, '_version'],
                     [_date_tag, '_date'],
                     [_description_tag, '_description']]

    _curr_host_marker = "<current-host>"

    class BadYamlError(Exception):
        def __init__(self, msg):
            super().__init__(msg)
            return

    @staticmethod
    def _curr_host(_: str = None) -> str:
        return socket.gethostbyname(socket.gethostname())

    def __init__(self,
                 settings_yaml_stream,
                 sections: Dict[str, List[str]] = None,
                 bespoke_transforms: List[Transformer.Transform] = None):
        """
        Boot strap the settings form the supplied YAML stream
        :param settings_yaml_stream: A callable that returns an open stream to the YAML source
        :param sections: List of List(s) of form ['section name as str','section item name', ...]
        :param bespoke_transforms: An optional list of transformers to be applied to settings
        """
        self._stream = None
        if sections is not None:
            if not isinstance(sections, Dict):
                raise ValueError("Settings sections must be type dictionary not {}".format(type(sections)))
            if len(sections) == 0:
                sections = None
        self._sections = sections

        self._transformer = Transformer()
        self._transformer.add_transform(Transformer.Transform(regular_expression=Settings._curr_host_marker,
                                                              transform=Settings._curr_host))
        if bespoke_transforms is not None:
            for transform in bespoke_transforms:
                self._transformer.add_transform(transform=transform)

        if settings_yaml_stream is None:
            raise ValueError(
                "Mandatory parameter YAML stream was passed as None")

        if not hasattr(settings_yaml_stream, "__call__"):
            raise ValueError(
                "YAML stream sources must be callable, [{}} is not callable".format(type(settings_yaml_stream)))

        # Header
        self._version = None
        self._date = None
        self._description = None

        self._yaml_stream = settings_yaml_stream
        self._load_settings()
        return

    def __del__(self):
        if hasattr(self, "_stream"):
            if self._stream is not None:
                self._stream.close()
        return

    @property
    def description(self) -> str:
        return self._description

    @property
    def version(self) -> str:
        return self._version

    @property
    def date(self) -> datetime:
        return datetime.strptime(self._date, "%d %b %Y")

    def _load_settings(self) -> None:
        """
        Load the settings from YAML config.
        """
        self._stream = self._yaml_stream()
        try:
            yml_map = yaml.safe_load(self._stream)
        except Exception as e:
            raise ValueError("Bad stream, could not load yaml from stream with exception [{}]".format(str(e)))

        if yml_map is None:
            raise Settings.BadYamlError(msg="supplied Yml stream contains no parsable yaml")

        self._parse_header(yml_map)
        self._parse_sections(yml_map)
        self._stream.close()
        return

    def _parse_header(self,
                      yml_map: Dict) -> None:
        """
        Extract the header details from the header section of the yaml
        :param yml_map: The parsed Yaml as a dictionary
        """
        if Settings._header not in yml_map:
            raise Settings.BadYamlError(
                msg="Mal-structured setting yaml [{}:] section is missing from header".format(Settings._header))

        header = yml_map.get(Settings._header)
        for item in Settings._header_items:
            if item[0] not in header:
                raise Settings.BadYamlError(
                    msg="Mal-structured setting yaml [{}] is missing from header".format(item[0]))
            setattr(self, item[1], header[item[0]])
        return

    def _get_section(self,
                     section_name: str) -> List[str]:
        """
        Return all the items for the given section - always in the order they were specified in the section
        desriptior passed to the __inti__ function
        :param section_name: The name of the section to get the items for
        :return:
        """
        section = self._sections[section_name]
        items = list()
        for item in section:
            items.append(getattr(self, "{}_{}".format(section_name, item)))
        return items

    def _parse_sections(self,
                        yml_map: Dict) -> None:
        """
        parse the yaml free or structured
        :param yml_map: The parsed Yaml as a dictionary
        """
        if self._sections is None:
            self._parse_sections_free_form(yml_map=yml_map)
        else:
            self._parse_sections_by_section(yml_map=yml_map)
        return

    def _parse_sections_by_section(self,
                                   yml_map: Dict) -> None:
        """
        Extract the each section detail and add members and accessor method. If YAML does not agree with the
        section spec throw and error.
        :param yml_map: The parsed Yaml as a dictionary

         Note: Below We need lambda in lambda here to force the argument to the final lambda function to be in its
         own scope - else every function will just return the items of the last sections defined.
         see: https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        """

        for section_name, section in self._sections.items():
            if not isinstance(section, List) or len(section) < 2:
                raise ValueError("section descriptor must be a Tuple of min length 2 items")

            if section_name not in yml_map:
                raise Settings.BadYamlError(
                    msg="Mal-structured yaml [{}:] section is missing from header".format(Settings._header))

            yaml_section = yml_map.get(section_name)
            for item in section:
                if item not in yaml_section:
                    raise Settings.BadYamlError(
                        msg="Mal-structured yaml [{}] section value is missing from header".format(item[0]))
                yaml_section[item] = self._transformer.transform(string_to_transform=yaml_section[item])
                # Dynamically add a member with name section_item
                setattr(self,
                        "{}_{}".format(section_name, item),
                        yaml_section[item])
            # Dynamically add a function called the same as the section name that will return the
            # section items as a list.
            setattr(self,
                    section_name,
                    [(lambda x: (lambda: x))(self._get_section(s)) for s in [section_name]][0])
        return

    def _parse_sections_free_form(self,
                                  yml_map: Dict) -> None:
        """
        Extract the YAML as it is defined and do not impose any structure/presence rules.
        :param yml_map: The parsed Yaml as a dictionary

         Note: Below We need lambda in lambda here to force the argument to the final lambda function to be in its
         own scope - else every function will just return the items of the last sections defined.
         see: https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        """
        self._sections = dict()
        for section_name, section in yml_map.items():
            if section_name != Settings._header:
                self._sections[section_name] = list()
                section = list(section.keys())
                yaml_section = yml_map.get(section_name)
                for item in section:
                    yaml_section[item] = self._transformer.transform(string_to_transform=yaml_section[item])
                    # Dynamically add a member with name section_item
                    setattr(self,
                            "{}_{}".format(section_name, item),
                            yaml_section[item])
                    self._sections[section_name].append(item)
                # Dynamically add a function called the same as the section name that will return the
                # section items as a list.
                setattr(self,
                        section_name,
                        [(lambda x: (lambda: x))(self._get_section(s)) for s in [section_name]][0])
        return
