<template>
  <el-menu
    :default-active="onRoutes"
    class="sidebar-content"
    unique-opened
    router
    @select="selectMenu">
    <sub-tree :tree="menu"/>
  </el-menu>
</template>
<script>
import subTree from './subTree.vue'
import { createNamespacedHelpers } from 'vuex'
const { mapState, mapMutations } = createNamespacedHelpers('sidebar')

export default {
  components: {
    subTree
  },
  computed: {
    onRoutes () {
      return this.$route.path.replace('/', '')
    },
    ...mapState(['menu', 'titlePath', 'routerPath'])
  },
  mounted () {
    this.routerPathClear()
    this.titlePathClear()
    this.initIndexPath(this.onRoutes, this.menu)
    this.pathToTitle(this.menu)
  },
  methods: {
    ...mapMutations([
      'routerPathPop',
      'routerPathPush',
      'routerPathUpdate',
      'routerPathClear',
      'titlePathPush',
      'titlePathClear'
    ]),
    /* 初始化菜单路径 */
    initIndexPath (router = '', menu = []) {
      for (let item of menu) {
        if (!item.subs && router === item.index) {
          this.routerPathPush(router)
          return true
        }
        if (item.subs) {
          this.routerPath.push(item.index)
          if (!this.initIndexPath(router, item.subs)) {
            this.routerPathPop()
          } else {
            return true
          }
        }
      }
    },
    /* 初始化菜单路径 - router 转换 title */
    pathToTitle (menu = []) {
      for (let item of menu) {
        if (this.routerPath.indexOf(item.index) !== -1) {
          this.titlePathPush(item.title)
          if (item.subs) {
            this.pathToTitle(item.subs)
          }
        }
      }
    },
    selectMenu (index, indexPath) {
      this.routerPathClear()
      this.routerPathUpdate(indexPath)
      this.titlePathClear()
      this.pathToTitle(this.menu)
    }
  }
}
</script>
<style lang="scss">
.sidebar-content {
  height: 100%;
}
</style>
