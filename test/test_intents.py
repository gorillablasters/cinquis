# importing module
import sys
# appending a path
sys.path.append('../src')
import cinquis.intents as ci


def main():
    print('test_intents')
    varNumcorpus = 'Corpus/varNum.txt'
    varStringcorpus = 'Corpus/varStr.txt'
    varVarcorpus = 'Corpus/varVar.txt'
    ci.addIntent("Num", varNumcorpus, "var number assignment")
    ci.addIntent("Str", varStringcorpus, "var string assignment")
    ci.addIntent("Var", varVarcorpus, "var var assignment")
    ci.createIntents('intents.json')
    print('test complete')

    #if len(sys.argv) > 3:
        #if(sys.argv[1]) == 'make_intent_from_file':
            #make_intent_from_file(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])


#def make_intent_from_file(tag,filepath,response,intentsFilePath):
    #it.addIntent(tag,filepath,response)
    #it.createIntents(intentsFilePath)



if __name__ == "__main__":
    main()
