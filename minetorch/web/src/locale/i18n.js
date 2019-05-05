import Vue from 'vue'
import VueI18n from 'vue-i18n'
import zhCN from './zh-CN'
import en from './en'
import enLocale from 'element-ui/lib/locale/lang/en'
import zhCNLocale from 'element-ui/lib/locale/lang/zh-CN'
import { getQueryVariable } from 'utils/tools'

Vue.use(VueI18n)

const messages = {
  en: Object.assign(en, enLocale),
  sc: Object.assign(zhCN, zhCNLocale)

}
let lang = getQueryVariable('lang') || window.localStorage.getItem('lang')

if (!lang) {
  lang = 'sc'
}
window.localStorage.setItem('lang', lang)

const i18n = new VueI18n({
  locale: localStorage.getItem('locale') || 'sc',
  messages
})

export default i18n
