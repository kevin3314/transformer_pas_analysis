from typing import List

from pyknp import Tag

from kwdlc_reader import KWDLCReader, Mention, Entity


def test_pas(fixture_kwdlc_readers: List[KWDLCReader]):
    kwdlc_reader = fixture_kwdlc_readers[0]
    predicates: List[Tag] = kwdlc_reader.get_predicates()
    assert len(predicates) == 11

    sid1 = 'w201106-0000060050-1'
    sid2 = 'w201106-0000060050-2'
    sid3 = 'w201106-0000060050-3'

    arguments = kwdlc_reader.get_arguments(predicates[0])
    assert predicates[0].midasi == 'トスを'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    assert tuple(arguments['ヲ'][0]) == (sid1, 0, 'コイン', 0, 'dep', '')

    arguments = kwdlc_reader.get_arguments(predicates[1])
    assert predicates[1].midasi == '行う。'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    # assert tuple(arguments['ヲ'][0]) == (sid1, 1, 'トス', 1, 'overt', '')  # 最新版はこっち
    assert tuple(arguments['ヲ'][0]) == (sid1, 1, 'コイントス', 1, 'overt', '')

    arguments = kwdlc_reader.get_arguments(predicates[2])
    assert predicates[2].midasi == '表が'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ノ'][0]) == (sid1, 0, 'コイン', 0, 'inter', '')

    arguments = kwdlc_reader.get_arguments(predicates[3])
    assert predicates[3].midasi == '出た'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (sid2, 0, '表', 4, 'overt', '')
    assert tuple(arguments['外の関係'][0]) == (sid2, 2, '数', 6, 'dep', '')

    arguments = kwdlc_reader.get_arguments(predicates[4])
    assert predicates[4].midasi == '数だけ、'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ノ'][0]) == (sid2, 1, '出た', 5, 'dep', '')

    arguments = kwdlc_reader.get_arguments(predicates[5])
    assert predicates[5].midasi == 'モンスターを'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['修飾'][0]) == (sid2, 3, 'フィールド上', 7, 'dep', '')
    assert tuple(arguments['修飾'][1]) == (sid2, 2, '数', 6, 'intra', 'AND')

    arguments = kwdlc_reader.get_arguments(predicates[6])
    assert predicates[6].midasi == '破壊する。'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:状況', None, 'exo', '')
    assert tuple(arguments['ヲ'][0]) == (sid2, 4, 'モンスター', 8, 'overt', '')

    arguments = kwdlc_reader.get_arguments(predicates[7])
    assert predicates[7].midasi == '効果は'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['トイウ'][0]) == (sid2, 5, '破壊する', 9, 'inter', '')

    arguments = kwdlc_reader.get_arguments(predicates[8])
    assert predicates[8].midasi == '１度だけ'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ニ'][0]) == (sid3, 3, 'ターン', 13, 'overt', '')

    # arguments = kwdlc_reader.get_arguments(predicates[9])
    # assert predicates[9].midasi == 'メイン'
    # assert len([_ for args in arguments.values() for _ in args]) == 1
    # assert tuple(arguments['ガ'][0]) == (sid3, 7, 'フェイズ', 17, 'dep', '')
    #
    # arguments = kwdlc_reader.get_arguments(predicates[10])
    # assert predicates[10].midasi == 'フェイズに'
    # assert len([_ for args in arguments.values() for _ in args]) == 1
    # assert tuple(arguments['ノ？'][0]) == (sid3, 5, '自分', 15, 'overt', '')
    #
    # arguments = kwdlc_reader.get_arguments(predicates[11])
    # assert predicates[11].midasi == '使用する事ができる。'
    # assert len([_ for args in arguments.values() for _ in args]) == 3
    # assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    # assert tuple(arguments['ヲ'][0]) == (sid3, 1, '効果', 11, 'dep', '')
    # assert tuple(arguments['ニ'][0]) == (sid3, 7, 'フェイズ', 17, 'overt', '')

    arguments = kwdlc_reader.get_arguments(predicates[9])
    assert predicates[9].midasi == 'メインフェイズに'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ノ？'][0]) == (sid3, 5, '自分', 15, 'overt', '')

    arguments = kwdlc_reader.get_arguments(predicates[10])
    assert predicates[10].midasi == '使用する事ができる。'
    assert len([_ for args in arguments.values() for _ in args]) == 3
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    assert tuple(arguments['ヲ'][0]) == (sid3, 1, '効果', 11, 'dep', '')
    assert tuple(arguments['ニ'][0]) == (sid3, 6, 'メインフェイズ', 16, 'overt', '')


def test_coref(fixture_kwdlc_readers: List[KWDLCReader]):
    kwdlc_reader = fixture_kwdlc_readers[1]
    entities: List[Entity] = kwdlc_reader.get_all_entities()
    assert len(entities) == 1

    entity: Entity = entities[0]
    mentions: List[Mention] = entity.mentions
    assert len(mentions) == 4
    assert (mentions[0].midasi, mentions[0].dtid) == ('ドクターを', 7)
    assert (mentions[1].midasi, mentions[1].dtid) == ('ドクターを', 11)
    assert (mentions[2].midasi, mentions[2].dtid) == ('ドクターの', 16)
    assert (mentions[3].midasi, mentions[3].dtid) == ('皆様', 17)
