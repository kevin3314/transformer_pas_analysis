from typing import List

from pyknp import Tag

from kwdlc_reader import KWDLCReader, Mention, Entity


def test_pas(fixture_kwdlc_reader: KWDLCReader):
    document = fixture_kwdlc_reader.process_document('w201106-0000060050')
    predicates: List[Tag] = document.get_predicates()
    assert len(predicates) == 11

    sid1 = 'w201106-0000060050-1'
    sid2 = 'w201106-0000060050-2'
    sid3 = 'w201106-0000060050-3'

    arguments = document.get_arguments(predicates[0])
    assert predicates[0].midasi == 'トスを'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    assert tuple(arguments['ヲ'][0]) == (sid1, 0, 'コイン', 0, 'dep', '')

    arguments = document.get_arguments(predicates[1])
    assert predicates[1].midasi == '行う。'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    # assert tuple(arguments['ヲ'][0]) == (sid1, 1, 'トス', 1, 'overt', '')  # 最新版はこっち
    assert tuple(arguments['ヲ'][0]) == (sid1, 1, 'コイントス', 1, 'overt', '')

    arguments = document.get_arguments(predicates[2])
    assert predicates[2].midasi == '表が'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ノ'][0]) == (sid1, 0, 'コイン', 0, 'inter', '')

    arguments = document.get_arguments(predicates[3])
    assert predicates[3].midasi == '出た'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (sid2, 0, '表', 4, 'overt', '')
    assert tuple(arguments['外の関係'][0]) == (sid2, 2, '数', 6, 'dep', '')

    arguments = document.get_arguments(predicates[4])
    assert predicates[4].midasi == '数だけ、'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ノ'][0]) == (sid2, 1, '出た', 5, 'dep', '')

    arguments = document.get_arguments(predicates[5])
    assert predicates[5].midasi == 'モンスターを'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['修飾'][0]) == (sid2, 3, 'フィールド上', 7, 'dep', '')
    assert tuple(arguments['修飾'][1]) == (sid2, 2, '数', 6, 'intra', 'AND')

    arguments = document.get_arguments(predicates[6])
    assert predicates[6].midasi == '破壊する。'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:状況', None, 'exo', '')
    assert tuple(arguments['ヲ'][0]) == (sid2, 4, 'モンスター', 8, 'overt', '')

    arguments = document.get_arguments(predicates[7])
    assert predicates[7].midasi == '効果は'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['トイウ'][0]) == (sid2, 5, '破壊する', 9, 'inter', '')

    arguments = document.get_arguments(predicates[8])
    assert predicates[8].midasi == '１度だけ'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ニ'][0]) == (sid3, 3, 'ターン', 13, 'overt', '')

    # arguments = document.get_arguments(predicates[9])
    # assert predicates[9].midasi == 'メイン'
    # assert len([_ for args in arguments.values() for _ in args]) == 1
    # assert tuple(arguments['ガ'][0]) == (sid3, 7, 'フェイズ', 17, 'dep', '')
    #
    # arguments = document.get_arguments(predicates[10])
    # assert predicates[10].midasi == 'フェイズに'
    # assert len([_ for args in arguments.values() for _ in args]) == 1
    # assert tuple(arguments['ノ？'][0]) == (sid3, 5, '自分', 15, 'overt', '')
    #
    # arguments = document.get_arguments(predicates[11])
    # assert predicates[11].midasi == '使用する事ができる。'
    # assert len([_ for args in arguments.values() for _ in args]) == 3
    # assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    # assert tuple(arguments['ヲ'][0]) == (sid3, 1, '効果', 11, 'dep', '')
    # assert tuple(arguments['ニ'][0]) == (sid3, 7, 'フェイズ', 17, 'overt', '')

    arguments = document.get_arguments(predicates[9])
    assert predicates[9].midasi == 'メインフェイズに'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    assert tuple(arguments['ノ？'][0]) == (sid3, 5, '自分', 15, 'overt', '')

    arguments = document.get_arguments(predicates[10])
    assert predicates[10].midasi == '使用する事ができる。'
    assert len([_ for args in arguments.values() for _ in args]) == 3
    assert tuple(arguments['ガ'][0]) == (None, None, '不特定:人', None, 'exo', '')
    assert tuple(arguments['ヲ'][0]) == (sid3, 1, '効果', 11, 'dep', '')
    assert tuple(arguments['ニ'][0]) == (sid3, 6, 'メインフェイズ', 16, 'overt', '')


def test_pas_relax(fixture_kwdlc_reader: KWDLCReader):
    document = fixture_kwdlc_reader.process_document('w201106-0000060050')
    predicates: List[Tag] = document.get_predicates()
    arguments = document.get_arguments(predicates[9], relax=True)
    sid3 = 'w201106-0000060050-3'
    assert predicates[9].midasi == 'メインフェイズに'
    assert len([_ for args in arguments.values() for _ in args]) == 4
    assert tuple(arguments['ノ？'][0]) == (sid3, 5, '自分', 15, 'overt', '')
    assert tuple(arguments['ノ？'][1]) == (None, None, '不特定:人', None, 'exo', 'AND')
    assert tuple(arguments['ノ？'][2]) == (None, None, '著者', None, 'exo', 'AND')
    assert tuple(arguments['ノ？'][3]) == (None, None, '読者', None, 'exo', 'AND')


def test_coref(fixture_kwdlc_reader: KWDLCReader):
    document = fixture_kwdlc_reader.process_document('w201106-0000060050')
    entities: List[Entity] = document.get_all_entities()
    assert len(entities) == 1

    entity: Entity = entities[0]
    assert (entity.taigen, entity.yougen) == (True, False)
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == 'AND'

    document = fixture_kwdlc_reader.process_document('w201106-0000060560')
    entities: List[Entity] = document.get_all_entities()
    assert len(entities) == 1

    entity: Entity = entities[0]
    assert (entity.taigen, entity.yougen) == (True, False)
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 4
    assert (mentions[0].midasi, mentions[0].dtid) == ('ドクターを', 7)
    assert (mentions[1].midasi, mentions[1].dtid) == ('ドクターを', 11)
    assert (mentions[2].midasi, mentions[2].dtid) == ('ドクターの', 16)
    assert (mentions[3].midasi, mentions[3].dtid) == ('皆様', 17)


def test_ne(fixture_kwdlc_reader: KWDLCReader):
    document = fixture_kwdlc_reader.process_document('w201106-0000060877')
    nes = document.named_entities
    assert len(nes) == 2
    ne = nes[0]
    assert (ne.category, ne.midasi, ne.dmid_range) == ('ORGANIZATION', '柏市ひまわり園', range(5, 9))
    ne = nes[1]
    assert (ne.category, ne.midasi, ne.dmid_range) == ('DATE', '平成２３年度', range(11, 14))

    document = fixture_kwdlc_reader.process_document('w201106-0000074273')
    nes = document.named_entities
    assert len(nes) == 3
    ne = nes[0]
    assert (ne.category, ne.midasi, ne.dmid_range) == ('LOCATION', 'ダーマ神殿', range(15, 17))
    ne = nes[1]
    assert (ne.category, ne.midasi, ne.dmid_range) == ('ARTIFACT', '天の箱舟', range(24, 27))
    ne = nes[2]
    assert (ne.category, ne.midasi, ne.dmid_range) == ('LOCATION', 'ナザム村', range(39, 41))
