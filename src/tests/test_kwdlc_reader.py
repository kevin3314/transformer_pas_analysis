from typing import List

from pyknp import Tag

from kwdlc_reader import KWDLCReader, Mention, Entity, Predicate, SpecialArgument, Argument


def test_pas(fixture_kwdlc_reader: KWDLCReader):
    document = fixture_kwdlc_reader.process_document('w201106-0000060050')
    predicates: List[Predicate] = document.get_predicates()
    assert len(predicates) == 12

    sid1 = 'w201106-0000060050-1'
    sid2 = 'w201106-0000060050-2'
    sid3 = 'w201106-0000060050-3'

    arguments = document.get_arguments(predicates[0])
    assert predicates[0].midasi == 'トスを'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    arg = arguments['ガ'][0]
    assert isinstance(arg, SpecialArgument)
    assert arg.midasi == '不特定:人'
    arg = arguments['ヲ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid1, 0, 'コイン', 0, 'dep', '')

    arguments = document.get_arguments(predicates[1])
    assert predicates[1].midasi == '行う。'
    assert len([_ for args in arguments.values() for _ in args]) == 4
    arg = arguments['ガ'][0]
    assert isinstance(arg, SpecialArgument)
    assert arg.midasi == '不特定:人'
    arg = arguments['ヲ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid1, 1, 'トス', 1, 'overt', '')

    arguments = document.get_arguments(predicates[2])
    assert predicates[2].midasi == '表が'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    arg = arguments['ノ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid1, 0, 'コイン', 0, 'inter', '')

    arguments = document.get_arguments(predicates[3])
    assert predicates[3].midasi == '出た'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    arg = arguments['ガ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid2, 0, '表', 4, 'overt', '')
    arg = arguments['外の関係'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid2, 2, '数', 6, 'dep', '')

    arguments = document.get_arguments(predicates[4])
    assert predicates[4].midasi == '数だけ、'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    arg = arguments['ノ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid2, 1, '出た', 5, 'dep', '')

    arguments = document.get_arguments(predicates[5])
    assert predicates[5].midasi == 'モンスターを'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    arg = arguments['修飾'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid2, 3, 'フィールド上', 7, 'dep', '')
    arg = arguments['修飾'][1]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid2, 2, '数', 6, 'intra', 'AND')

    arguments = document.get_arguments(predicates[6])
    assert predicates[6].midasi == '破壊する。'
    assert len([_ for args in arguments.values() for _ in args]) == 2
    arg = arguments['ガ'][0]
    assert isinstance(arg, SpecialArgument)
    assert arg.midasi == '不特定:状況'
    arg = arguments['ヲ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid2, 4, 'モンスター', 8, 'overt', '')

    arguments = document.get_arguments(predicates[7])
    assert predicates[7].midasi == '効果は'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    arg = arguments['トイウ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid2, 5, '破壊する', 9, 'inter', '')

    arguments = document.get_arguments(predicates[8])
    assert predicates[8].midasi == '１度だけ'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    arg = arguments['ニ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid3, 3, 'ターン', 13, 'overt', '')

    arguments = document.get_arguments(predicates[9])
    assert predicates[9].midasi == 'メイン'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    arg = arguments['ガ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid3, 7, 'フェイズ', 17, 'dep', '')

    arguments = document.get_arguments(predicates[10])
    assert predicates[10].midasi == 'フェイズに'
    assert len([_ for args in arguments.values() for _ in args]) == 1
    arg = arguments['ノ？'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid3, 5, '自分', 15, 'overt', '')

    arguments = document.get_arguments(predicates[11])
    assert predicates[11].midasi == '使用する事ができる。'
    assert len([_ for args in arguments.values() for _ in args]) == 5
    arg = arguments['ガ'][0]
    assert isinstance(arg, SpecialArgument)
    assert arg.midasi == '不特定:人'
    arg = arguments['ガ'][1]
    assert isinstance(arg, SpecialArgument)
    assert arg.midasi == '著者'
    arg = arguments['ガ'][2]
    assert isinstance(arg, SpecialArgument)
    assert arg.midasi == '読者'
    arg = arguments['ヲ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid3, 1, '効果', 11, 'dep', '')
    arg = arguments['ニ'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid3, 7, 'フェイズ', 17, 'overt', '')


def test_pas_relax(fixture_kwdlc_reader: KWDLCReader):
    document = fixture_kwdlc_reader.process_document('w201106-0000060050')
    predicates: List[Predicate] = document.get_predicates()
    arguments = document.get_arguments(predicates[10], relax=True)
    sid3 = 'w201106-0000060050-3'
    assert predicates[10].midasi == 'フェイズに'
    assert len([_ for args in arguments.values() for _ in args]) == 4
    arg = arguments['ノ？'][0]
    assert isinstance(arg, Argument)
    assert tuple(arg) == (sid3, 5, '自分', 15, 'overt', '')
    arg = arguments['ノ？'][1]
    assert isinstance(arg, SpecialArgument)
    assert arg.midasi == '不特定:人'
    arg = arguments['ノ？'][2]
    assert isinstance(arg, SpecialArgument)
    assert (arg.midasi, arg.dep_type, arg.mode) == ('著者', 'exo', 'AND')
    arg = arguments['ノ？'][3]
    assert isinstance(arg, SpecialArgument)
    assert (arg.midasi, arg.dep_type, arg.mode) == ('読者', 'exo', 'AND')


def test_coref(fixture_kwdlc_reader: KWDLCReader):
    document = fixture_kwdlc_reader.process_document('w201106-0000060050')
    entities: List[Entity] = document.get_all_entities()
    assert len(entities) == 16

    entity = entities[0]
    assert (entity.taigen, entity.yougen) == (True, True)  # TODO: should be (False, False)
    assert entity.exophors == ['不特定:人']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 0

    entity = entities[1]
    assert (entity.taigen, entity.yougen) == (True, False)
    assert entity.exophors == []
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('コイン', 0)
    assert entity.exophors == []
    assert entity.mode == ''

    entity = entities[2]
    assert (entity.taigen, entity.yougen) == (True, True)  # TODO: should be (False, False)
    assert entity.exophors == ['不特定:人']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 0

    entity = entities[3]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[4]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[5]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[6]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[7]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[8]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[9]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[10]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

    entity = entities[11]
    assert (entity.taigen, entity.yougen) == (True, True)
    assert entity.exophors == ['不特定:人', '著者', '読者']
    assert entity.mode == ''
    mentions: List[Mention] = sorted(entity.mentions, key=lambda x: x.dtid)
    assert len(mentions) == 1
    assert (mentions[0].midasi, mentions[0].dtid) == ('自分の', 15)

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
